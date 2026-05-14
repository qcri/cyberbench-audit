"""Step 0 — Judge-agreement analysis (run before results_table.py).

Quantifies how much the three judge tracks agree:
  - default  (judge_<MODEL>/eval_results/)        — current GPT-5.4 unified prompt
  - v1       (judge_<MODEL>_v1/eval_results/)     — older judge prompt
  - v2       (judge_<MODEL>_v2/eval_results/)     — re-judged on upstream inference

Default and v1 score the same `responses_<MODEL>/` set, so per-sample agreement
(Cohen's κ, raw % agreement) is meaningful for that pair. v2 scores a different
inference run, so per-sample κ between v2 and the others is skipped — instead
we compare aggregate cell accuracy.

Outputs (under analysis/reports/judge_agreement/):
  - per_cell_default_vs_v1.csv     κ, raw_agreement, n, n_disagree per (model, task)
  - per_cell_aggregate_compare.csv accuracy(default/v1/v2) per (model, task)
  - summary.md                     overall κ averages, per-task κ, top disagreements
  - disagreements/<model>__<task>.jsonl   sample-level rows where def disagrees
                                          with v1 (capped at TOP_DISAGREE_PER_CELL)

Run from BenchmarkingSecBenchmarks/:
  PYTHONPATH=. python -m analysis.judge_agreement
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

from analysis.lib.loaders import (
    PARENT_GROUPS,
    TASK_ORDER,
    detailed_path,
    discover_models,
    iter_jsonl,
    load_summary_accuracy,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
REPORTS_DIR = HERE / "reports" / "judge_agreement"
DISAGREE_DIR = REPORTS_DIR / "disagreements"

TOP_DISAGREE_PER_CELL = 25  # cap how many disagreement samples we dump


# ─────────────────────────────────────────────────────────────────────────────
# Per-judge sample loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_records(model: str, task: str, version: str) -> List[dict]:
    path = detailed_path(OUTPUTS_ROOT, model, task, version=version)
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def _derive_is_correct(r: dict) -> Optional[bool]:
    """Map a per-sample record to a binary correctness flag.

    The default judge stores `is_correct`. The v1 judge sometimes uses
    task-specific fields:
      - `exact_match` for ATE / athena_ate (set equality) — same semantics as
        is_correct for that task type.
      - `mad` for VSP / athena_vsp (Manhattan distance between predicted and
        gold CVSS vectors). MAD == 0 means exact-match correctness.

    Returns None if no binary signal can be derived.
    """
    if r.get("is_correct") is not None:
        return bool(r["is_correct"])
    if r.get("exact_match") is not None:
        return bool(r["exact_match"])
    mad = r.get("mad")
    if mad is not None:
        try:
            return float(mad) == 0.0
        except (TypeError, ValueError):
            pass
    return None


def _index_by_idx(records: List[dict]) -> Dict[str, dict]:
    """Map sample index → record (with binary correctness derived).

    Records are augmented with a synthetic `_is_correct_binary` key so callers
    can compare across schemas. Skipped samples and ones with no binary signal
    are dropped.
    """
    out: Dict[str, dict] = {}
    for r in records:
        if r.get("skipped"):
            continue
        flag = _derive_is_correct(r)
        if flag is None:
            continue
        idx = str(r.get("index", ""))
        if idx == "":
            continue
        # mutate in place — records aren't reused outside this function
        r["_is_correct_binary"] = flag
        out[idx] = r
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cohen's κ
# ─────────────────────────────────────────────────────────────────────────────


def cohen_kappa(a: List[bool], b: List[bool]) -> Optional[float]:
    """Cohen's κ for two binary raters scoring the same N items.

    Returns None when N == 0 or when the expected agreement is 1 (degenerate
    perfect rater agreement on a constant label, where κ is undefined).
    """
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a_pos = sum(a) / n
    p_b_pos = sum(b) / n
    pe = p_a_pos * p_b_pos + (1 - p_a_pos) * (1 - p_b_pos)
    if pe >= 1.0 - 1e-12:
        # Perfectly degenerate (e.g., both raters always say correct). κ is
        # undefined but observed agreement is meaningful — return None and the
        # caller will report raw_agreement instead.
        return None
    return (po - pe) / (1 - pe)


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell comparisons
# ─────────────────────────────────────────────────────────────────────────────


def compare_default_vs_v1(model: str, task: str) -> Optional[dict]:
    """Per-sample comparison (alignable: same response set)."""
    rec_def = _index_by_idx(_load_records(model, task, ""))
    rec_v1 = _index_by_idx(_load_records(model, task, "_v1"))
    if not rec_def or not rec_v1:
        return None
    shared_idx = sorted(set(rec_def) & set(rec_v1), key=_idx_sortkey)
    if not shared_idx:
        return None
    a = [bool(rec_def[i]["_is_correct_binary"]) for i in shared_idx]
    b = [bool(rec_v1[i]["_is_correct_binary"]) for i in shared_idx]
    n = len(shared_idx)
    n_agree = sum(1 for x, y in zip(a, b) if x == y)
    raw = n_agree / n
    kappa = cohen_kappa(a, b)
    disagree_idx = [i for i, (x, y) in zip(shared_idx, zip(a, b)) if x != y]
    return {
        "n": n,
        "raw_agreement": raw,
        "kappa": kappa,
        "n_disagree": len(disagree_idx),
        "n_def_correct_v1_wrong": sum(1 for i, (x, y) in zip(shared_idx, zip(a, b)) if x and not y),
        "n_v1_correct_def_wrong": sum(1 for i, (x, y) in zip(shared_idx, zip(a, b)) if y and not x),
        "disagree_idx": disagree_idx,
        "rec_def": rec_def,
        "rec_v1": rec_v1,
    }


def aggregate_compare(model: str, task: str) -> dict:
    """Aggregate cell accuracy from each judge version.

    Uses summary.json when available; otherwise mean(is_correct) from the
    detailed file. Returns dict with keys 'def', 'v1', 'v2' (None when missing).
    """
    out = {}
    for ver, key in [("", "def"), ("_v1", "v1"), ("_v2", "v2")]:
        acc = load_summary_accuracy(OUTPUTS_ROOT, model, task, version=ver)
        n = 0
        if acc is None:
            recs = _index_by_idx(_load_records(model, task, ver))
            if recs:
                vals = [bool(r["_is_correct_binary"]) for r in recs.values()]
                acc = sum(vals) / len(vals)
                n = len(vals)
        else:
            recs = _index_by_idx(_load_records(model, task, ver))
            n = len(recs)
        out[key] = (acc, n)
    return out


def _idx_sortkey(idx: str):
    try:
        return (0, int(idx))
    except ValueError:
        return (1, idx)


# ─────────────────────────────────────────────────────────────────────────────
# Disagreement sample dump
# ─────────────────────────────────────────────────────────────────────────────


def dump_disagreements(model: str, task: str, comp: dict) -> Optional[Path]:
    if not comp["disagree_idx"]:
        return None
    DISAGREE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DISAGREE_DIR / f"{model}__{task}.jsonl"
    rows: List[dict] = []
    for idx in comp["disagree_idx"][:TOP_DISAGREE_PER_CELL]:
        d = comp["rec_def"][idx]
        v = comp["rec_v1"][idx]
        rows.append({
            "model": model,
            "task": task,
            "index": idx,
            "ground_truth": d.get("ground_truth", v.get("ground_truth", "")),
            "model_response": d.get("model_response", v.get("model_response", "")),
            "default": {
                "is_correct": d.get("_is_correct_binary"),
                "extracted_answer": d.get("extracted_answer") or d.get("extracted_vector", ""),
                "justification": d.get("judge_justification", ""),
            },
            "v1": {
                "is_correct": v.get("_is_correct_binary"),
                "extracted_answer": v.get("extracted_answer") or v.get("extracted_vector", ""),
                "justification": v.get("judge_justification", ""),
                "raw_signals": {k: v[k] for k in ("exact_match", "mad", "tp", "fp", "fn") if k in v},
            },
        })
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────────────


def write_per_cell_csvs(rows_dv1: List[dict], rows_agg: List[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORTS_DIR / "per_cell_default_vs_v1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "n", "raw_agreement", "kappa",
                    "n_disagree", "n_def_correct_v1_wrong", "n_v1_correct_def_wrong"])
        for r in rows_dv1:
            w.writerow([
                r["model"], r["task"], r["n"],
                f"{r['raw_agreement']:.4f}",
                "NA" if r["kappa"] is None else f"{r['kappa']:.4f}",
                r["n_disagree"], r["n_def_correct_v1_wrong"], r["n_v1_correct_def_wrong"],
            ])

    with open(REPORTS_DIR / "per_cell_aggregate_compare.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task",
                    "acc_default", "n_default",
                    "acc_v1", "n_v1",
                    "acc_v2", "n_v2"])
        for r in rows_agg:
            row = [r["model"], r["task"]]
            for k in ("def", "v1", "v2"):
                acc, n = r[k]
                row.extend(["NA" if acc is None else f"{acc:.4f}", n])
            w.writerow(row)


def write_summary_md(rows_dv1: List[dict], rows_agg: List[dict],
                     models: List[str]) -> None:
    lines: List[str] = []
    lines.append("# Judge-agreement summary")
    lines.append("")
    lines.append("Generated by `analysis/judge_agreement.py`. Three judge tracks:")
    lines.append("")
    lines.append("- **default** — `judge_<MODEL>/eval_results/`")
    lines.append("- **v1** — `judge_<MODEL>_v1/eval_results/`")
    lines.append("- **v2** — `judge_<MODEL>_v2/eval_results/` (different inference run)")
    lines.append("")
    lines.append("Default and v1 judge the same `responses_<MODEL>/` file so they "
                 "support per-sample κ. v2 judges `responses_<MODEL>_v2/` (a "
                 "different inference); we report only its aggregate cell accuracy.")
    lines.append("")

    # Overall κ across cells
    kappas = [r["kappa"] for r in rows_dv1 if r["kappa"] is not None]
    raws = [r["raw_agreement"] for r in rows_dv1]
    if kappas:
        lines.append(f"## Overall (default vs v1)")
        lines.append("")
        lines.append(f"- Cells compared: **{len(rows_dv1)}**")
        lines.append(f"- Mean κ: **{mean(kappas):.3f}**  (n={len(kappas)} non-degenerate)")
        lines.append(f"- Mean raw agreement: **{mean(raws):.3f}**")
        lines.append("")

    # Per-task κ (averaged across models)
    by_task: Dict[str, List[float]] = {}
    by_task_raw: Dict[str, List[float]] = {}
    for r in rows_dv1:
        if r["kappa"] is not None:
            by_task.setdefault(r["task"], []).append(r["kappa"])
        by_task_raw.setdefault(r["task"], []).append(r["raw_agreement"])

    lines.append("## Per-task agreement (default vs v1, averaged across models)")
    lines.append("")
    lines.append("| Task | mean κ | mean raw agreement | models |")
    lines.append("|---|---|---|---|")
    for task in TASK_ORDER:
        kvals = by_task.get(task, [])
        rvals = by_task_raw.get(task, [])
        kstr = f"{mean(kvals):.3f}" if kvals else "NA"
        rstr = f"{mean(rvals):.3f}" if rvals else "NA"
        n_models = len(rvals)
        lines.append(f"| {task} | {kstr} | {rstr} | {n_models} |")
    lines.append("")

    # Per-model κ (averaged across tasks)
    by_model: Dict[str, List[float]] = {}
    by_model_raw: Dict[str, List[float]] = {}
    for r in rows_dv1:
        if r["kappa"] is not None:
            by_model.setdefault(r["model"], []).append(r["kappa"])
        by_model_raw.setdefault(r["model"], []).append(r["raw_agreement"])

    lines.append("## Per-model agreement (default vs v1, averaged across tasks)")
    lines.append("")
    lines.append("| Model | mean κ | mean raw agreement | tasks |")
    lines.append("|---|---|---|---|")
    for m in models:
        kvals = by_model.get(m, [])
        rvals = by_model_raw.get(m, [])
        kstr = f"{mean(kvals):.3f}" if kvals else "NA"
        rstr = f"{mean(rvals):.3f}" if rvals else "NA"
        n_tasks = len(rvals)
        lines.append(f"| {m} | {kstr} | {rstr} | {n_tasks} |")
    lines.append("")

    # v2 coverage
    v2_rows = [r for r in rows_agg if r["v2"][0] is not None]
    lines.append("## v2 coverage")
    lines.append("")
    lines.append(f"v2 contributed to **{len(v2_rows)}** cells across "
                 f"**{len(set(r['model'] for r in v2_rows))}** models.")
    lines.append("")
    if v2_rows:
        # Find biggest discrepancies between v2 and the def/v1 average
        deltas = []
        for r in v2_rows:
            d, _ = r["def"]
            v1, _ = r["v1"]
            v2_acc, _ = r["v2"]
            base = [x for x in (d, v1) if x is not None]
            if not base:
                continue
            base_avg = mean(base)
            deltas.append((abs(v2_acc - base_avg), r["model"], r["task"],
                           v2_acc, base_avg))
        deltas.sort(reverse=True)
        lines.append("### Top v2 vs (def avg v1) discrepancies (|delta| ≥ 0.05)")
        lines.append("")
        lines.append("| model | task | v2 acc | def/v1 mean | |Δ| |")
        lines.append("|---|---|---|---|---|")
        for delta, m, t, v2_acc, base_avg in deltas[:20]:
            if delta < 0.05:
                break
            lines.append(f"| {m} | {t} | {v2_acc:.3f} | {base_avg:.3f} | {delta:.3f} |")
        lines.append("")

    # Top disagreements (lowest κ cells)
    finite = [r for r in rows_dv1 if r["kappa"] is not None]
    finite.sort(key=lambda r: r["kappa"])
    lines.append("## Cells with weakest default-vs-v1 agreement (lowest κ)")
    lines.append("")
    lines.append("| model | task | κ | raw | n | n_disagree |")
    lines.append("|---|---|---|---|---|---|")
    for r in finite[:20]:
        lines.append(f"| {r['model']} | {r['task']} | {r['kappa']:.3f} | "
                     f"{r['raw_agreement']:.3f} | {r['n']} | {r['n_disagree']} |")
    lines.append("")

    (REPORTS_DIR / "summary.md").write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    models = discover_models(OUTPUTS_ROOT)
    print(f"Models: {len(models)}  Tasks: {len(TASK_ORDER)}")

    rows_dv1: List[dict] = []
    rows_agg: List[dict] = []
    n_disagree_dumps = 0

    for m in models:
        for t in TASK_ORDER:
            comp = compare_default_vs_v1(m, t)
            if comp is not None:
                rows_dv1.append({"model": m, "task": t, **{k: comp[k] for k in (
                    "n", "raw_agreement", "kappa", "n_disagree",
                    "n_def_correct_v1_wrong", "n_v1_correct_def_wrong",
                )}})
                if comp["disagree_idx"]:
                    if dump_disagreements(m, t, comp) is not None:
                        n_disagree_dumps += 1

            agg = aggregate_compare(m, t)
            # Only emit aggregate row if at least one judge has data
            if any(agg[k][0] is not None for k in ("def", "v1", "v2")):
                rows_agg.append({"model": m, "task": t, **agg})

    write_per_cell_csvs(rows_dv1, rows_agg)
    write_summary_md(rows_dv1, rows_agg, models)

    print(f"Wrote {REPORTS_DIR}/")
    print(f"  per_cell_default_vs_v1.csv    — {len(rows_dv1)} rows")
    print(f"  per_cell_aggregate_compare.csv — {len(rows_agg)} rows")
    print(f"  summary.md")
    print(f"  disagreements/                 — {n_disagree_dumps} JSONL files")


if __name__ == "__main__":
    main()
