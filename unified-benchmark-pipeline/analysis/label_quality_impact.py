"""How does removing label-quality-suspect samples affect the master ranking?

Reads `analysis/reports/verification/verdicts/direct/<task>_<idx>.json` to
build an exclusion set of (task, index) pairs whose direct-verifier verdict is
NOT `gold_correct` — i.e. the gold label is mislabelled (`majority_correct`),
both gold and models are wrong (`both_wrong`), or the verifier is unsure
(`uncertain`). We then recompute per-(model, task) accuracy by skipping those
samples in `<task>_detailed.jsonl`, and compare per-model averages to the
baseline.

Outputs (under analysis/reports/label_quality_impact/):
  - results_table_filtered.csv          new per-cell accuracies
  - delta_table.csv                     per-model (Δ avg, baseline avg, filtered avg)
  - ranking_diff.md                     before/after rankings + commentary
  - excluded_samples.csv                the (task, index, verdict) we dropped
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Set, Tuple

from analysis.lib.loaders import (
    PARENT_GROUPS,
    TASK_ORDER,
    detailed_path,
    discover_models,
    iter_jsonl,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
VERDICTS_DIR = HERE / "reports" / "verification" / "verdicts" / "direct"
REPORTS_DIR = HERE / "reports" / "label_quality_impact"

# Verdict categories to drop from the denominator. `gold_correct` is kept
# (those flags are false positives — gold is right and models are wrong, so
# the cell should still penalise the model).
DROP_VERDICTS = {"majority_correct", "both_wrong", "uncertain"}


def load_excluded() -> Tuple[Set[Tuple[str, str]], list]:
    """Return ((task, idx_str) set to skip, list of audit rows)."""
    excluded: Set[Tuple[str, str]] = set()
    audit = []
    if not VERDICTS_DIR.is_dir():
        return excluded, audit
    for f in sorted(VERDICTS_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        verdict = d.get("verdict", "")
        task = d.get("task", "")
        idx = str(d.get("index", ""))
        if not task or not idx:
            continue
        audit.append({
            "task": task, "index": idx,
            "verdict": verdict,
            "gold": d.get("gold", ""),
            "majority_prediction": d.get("majority_prediction", ""),
            "agreement_fraction": d.get("agreement_fraction", ""),
            "excluded": verdict in DROP_VERDICTS,
        })
        if verdict in DROP_VERDICTS:
            excluded.add((task, idx))
    return excluded, audit


def cell_accuracy_filtered(model: str, task: str,
                           excluded: Set[Tuple[str, str]]) -> Tuple[float | None, int, int]:
    """Recompute accuracy from detailed.jsonl, skipping excluded indices.

    Returns (acc, n_evaluated_after_filter, n_dropped).
    """
    p = detailed_path(OUTPUTS_ROOT, model, task)
    if not p.exists():
        return None, 0, 0
    correct = total = dropped = 0
    for r in iter_jsonl(p):
        if r.get("skipped"):
            continue
        if "is_correct" not in r:
            continue
        idx = str(r.get("index", ""))
        if (task, idx) in excluded:
            dropped += 1
            continue
        total += 1
        if r["is_correct"]:
            correct += 1
    return (correct / total if total > 0 else None), total, dropped


def cell_accuracy_baseline(model: str, task: str) -> Tuple[float | None, int]:
    p = detailed_path(OUTPUTS_ROOT, model, task)
    if not p.exists():
        return None, 0
    correct = total = 0
    for r in iter_jsonl(p):
        if r.get("skipped") or "is_correct" not in r:
            continue
        total += 1
        if r["is_correct"]:
            correct += 1
    return (correct / total if total > 0 else None), total


def fmt_pct(v):
    return "NA" if v is None else f"{v*100:.2f}"


def fmt_delta(v_after, v_before):
    if v_after is None or v_before is None:
        return "NA"
    d = (v_after - v_before) * 100
    return f"{'+' if d >= 0 else ''}{d:.2f}"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    excluded, audit = load_excluded()
    print(f"Loaded {len(audit)} verified samples; "
          f"{len(excluded)} flagged to exclude (verdicts: {DROP_VERDICTS}).")

    # Per-task counts of what we exclude
    by_task = {}
    for a in audit:
        if a["excluded"]:
            by_task.setdefault(a["task"], 0)
            by_task[a["task"]] += 1
    print("Per-task exclusions:")
    for t, n in sorted(by_task.items(), key=lambda x: -x[1]):
        print(f"  {t:<22s} {n}")

    models = discover_models(OUTPUTS_ROOT)

    # Per-cell baseline + filtered accuracies
    baseline = {m: {} for m in models}
    filtered = {m: {} for m in models}
    n_dropped_per_cell = {m: {} for m in models}
    for m in models:
        for t in TASK_ORDER:
            b, _ = cell_accuracy_baseline(m, t)
            baseline[m][t] = b
            f_acc, _, n_drop = cell_accuracy_filtered(m, t, excluded)
            filtered[m][t] = f_acc
            n_dropped_per_cell[m][t] = n_drop

    # Filtered results CSV
    with open(REPORTS_DIR / "results_table_filtered.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", *models])
        for t in TASK_ORDER:
            w.writerow([t, *(fmt_pct(filtered[m].get(t)) for m in models)])

    # Per-model averages — filtered vs baseline
    avg_baseline = {m: mean(v for v in baseline[m].values() if v is not None)
                    if any(v is not None for v in baseline[m].values()) else None
                    for m in models}
    avg_filtered = {m: mean(v for v in filtered[m].values() if v is not None)
                    if any(v is not None for v in filtered[m].values()) else None
                    for m in models}

    rank_baseline = sorted(models, key=lambda m: (avg_baseline[m] is None,
                                                   -(avg_baseline[m] or 0.0)))
    rank_filtered = sorted(models, key=lambda m: (avg_filtered[m] is None,
                                                   -(avg_filtered[m] or 0.0)))

    # Delta CSV
    with open(REPORTS_DIR / "delta_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "avg_baseline", "avg_filtered",
                    "delta_pct_pts", "rank_baseline", "rank_filtered", "rank_change"])
        for m in models:
            r_b = rank_baseline.index(m) + 1
            r_f = rank_filtered.index(m) + 1
            w.writerow([
                m,
                fmt_pct(avg_baseline[m]),
                fmt_pct(avg_filtered[m]),
                fmt_delta(avg_filtered[m], avg_baseline[m]),
                r_b, r_f, r_b - r_f,
            ])

    # Excluded samples CSV (audit trail)
    with open(REPORTS_DIR / "excluded_samples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "index", "verdict", "gold", "majority_prediction",
                    "agreement_fraction", "excluded"])
        for a in audit:
            w.writerow([a["task"], a["index"], a["verdict"], a["gold"],
                        a["majority_prediction"], a["agreement_fraction"],
                        "yes" if a["excluded"] else "no"])

    # Ranking diff markdown
    md = ["# Label-quality impact on the master ranking",
          "",
          f"Excluded **{len(excluded)} samples** "
          f"(verdicts {sorted(DROP_VERDICTS)} from the GPT-5.4 direct verifier "
          f"at $\\tau=0.75$) and recomputed per-(model, task) accuracy by skipping "
          f"those (task, index) pairs in `*_detailed.jsonl`.",
          "",
          "## Per-task exclusion counts",
          "",
          "| Task | Excluded |",
          "|---|---|"]
    for t, n in sorted(by_task.items(), key=lambda x: -x[1]):
        md.append(f"| {t} | {n} |")
    md.append("")

    md.append("## Ranking — before vs after")
    md.append("")
    md.append("| Rank | Baseline (full) | Filtered (exclude wrong+uncertain) |")
    md.append("|---|---|---|")
    for i in range(len(models)):
        b = rank_baseline[i] if i < len(rank_baseline) else ""
        ff = rank_filtered[i] if i < len(rank_filtered) else ""
        b_str = f"{b} ({fmt_pct(avg_baseline[b])})" if b else ""
        f_str = f"{ff} ({fmt_pct(avg_filtered[ff])})" if ff else ""
        md.append(f"| {i+1} | {b_str} | {f_str} |")
    md.append("")

    md.append("## Per-model deltas (sorted by |Δ|)")
    md.append("")
    md.append("| Model | Baseline avg (\\%) | Filtered avg (\\%) | Δ (pp) | Rank baseline → filtered |")
    md.append("|---|---|---|---|---|")
    sorted_by_delta = sorted(
        models,
        key=lambda m: -(abs((avg_filtered[m] or 0.0) - (avg_baseline[m] or 0.0)))
    )
    for m in sorted_by_delta:
        r_b = rank_baseline.index(m) + 1
        r_f = rank_filtered.index(m) + 1
        rc = "—" if r_b == r_f else (f"+{r_b - r_f}" if r_b > r_f else f"{r_b - r_f}")
        md.append(
            f"| {m} | {fmt_pct(avg_baseline[m])} | {fmt_pct(avg_filtered[m])} | "
            f"{fmt_delta(avg_filtered[m], avg_baseline[m])} | "
            f"{r_b} → {r_f} ({rc}) |"
        )
    md.append("")

    # Find any cells whose ordering swapped
    md.append("## Per-cell sanity")
    md.append("")
    md.append("Cells that gained the most accuracy after filtering "
              "(highest |Δ| at the cell level — these are the tasks where the "
              "removed samples were heavily biasing scores):")
    md.append("")
    md.append("| Model | Task | Baseline | Filtered | Δ (pp) | n_dropped |")
    md.append("|---|---|---|---|---|---|")
    cell_deltas = []
    for m in models:
        for t in TASK_ORDER:
            b = baseline[m].get(t)
            ff = filtered[m].get(t)
            if b is None or ff is None:
                continue
            d = (ff - b) * 100
            cell_deltas.append((abs(d), d, m, t, b, ff, n_dropped_per_cell[m][t]))
    cell_deltas.sort(reverse=True)
    for absd, d, m, t, b, ff, n in cell_deltas[:15]:
        md.append(f"| {m} | {t} | {fmt_pct(b)} | {fmt_pct(ff)} | "
                  f"{'+' if d >= 0 else ''}{d:.2f} | {n} |")
    md.append("")

    (REPORTS_DIR / "ranking_diff.md").write_text("\n".join(md))
    print(f"\nWrote {REPORTS_DIR}/{{results_table_filtered.csv, delta_table.csv, "
          f"ranking_diff.md, excluded_samples.csv}}")
    print(f"\nBaseline ranking:  {[m for m in rank_baseline]}")
    print(f"Filtered ranking:  {[m for m in rank_filtered]}")


if __name__ == "__main__":
    main()
