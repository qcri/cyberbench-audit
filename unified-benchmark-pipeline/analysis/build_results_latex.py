"""Generate the LaTeX `tab:main_results` block from results_table.csv.

Layout:
- Tasks as rows, models as columns (rotated 65°).
- Models sorted by best overall average (left = highest avg).
- Tasks grouped by parent benchmark (CTI, ATHENA, SECURE, REDSAGE,
  CYBERMETRIC, MCQ-Standalone, SEVENLLM) — same as PARENT_GROUPS.
- Cell value × 100, one decimal. **Bold** = best per task (highest).
- NA cells rendered as `--`.
- Last row: per-model average across populated cells.

Output: prints the full LaTeX block to stdout.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "reports" / "results_table.csv"
PROJECT_ROOT = HERE.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

PARENT_GROUPS = [
    # CTI-Bench (RISys-Lab HF cti-* + maveryn TSV cti-taa)
    ("CTI-Bench", ["mcq", "rcm", "vsp", "ate", "cti_taa"]),
    # AthenaBench (JSONL: CKT/RMS + 100-item TAA + the 3 expanded ATE/RCM/VSP)
    ("AthenaBench", ["ckt", "rms", "taa", "athena_ate", "athena_rcm", "athena_vsp"]),
    ("SECURE", ["secure_maet", "secure_cwet", "secure_kcv"]),
    ("RedSage-MCQ", ["redsage_frameworks", "redsage_generals", "redsage_skills",
                     "redsage_cli", "redsage_kali"]),
    ("CyberMetric", ["cybermetric"]),
    # MMLU-CS / SecBench / SecEval reported as separate single-task families.
    ("MMLU-CS", ["mmlu_cs"]),
    ("SecBench", ["secbench"]),
    ("SecEval", ["seceval"]),
    ("SEvenLLM", ["sevenllm"]),
]

# Display name mapping for tasks (collisions are OK — within-group only)
TASK_DISPLAY = {
    # CTI-Bench
    "mcq": "MCQ", "rcm": "RCM", "vsp": "VSP", "ate": "ATE", "cti_taa": "TAA",
    # AthenaBench
    "ckt": "CKT", "rms": "RMS", "taa": "TAA",
    "athena_ate": "ATE", "athena_rcm": "RCM", "athena_vsp": "VSP",
    # SECURE
    "secure_maet": "MAET", "secure_cwet": "CWET", "secure_kcv": "KCV",
    # RedSage
    "redsage_frameworks": "FW", "redsage_generals": "GEN",
    "redsage_skills": "Skills", "redsage_cli": "CLI", "redsage_kali": "Kali",
    # General
    "cybermetric": "CyberMetric", "mmlu_cs": "MMLU-CS",
    "secbench": "SecBench", "seceval": "SecEval",
    # SEvenLLM
    "sevenllm": "SEvenLLM",
}

# Per-task metric spec: (label_for_superscript, kind, lower_is_better)
#   kind ∈ {"acc", "f1", "mad", "mad_norm"}
#   "acc": strict-verdict accuracy (×100, %)
#   "f1":  F1 over the extracted-vs-gold ID set (×100, %)
#   "mad": mean absolute deviation of CVSS base score (raw, lower is better)
#   "mad_norm": max(0, 1 - mad/7.7) × 100 (AthenaBench's normalised MAD-acc)
TASK_METRIC = {t: (r"\textsuperscript{Acc}", "acc", False) for t in TASK_DISPLAY}
TASK_METRIC["vsp"]        = (r"\textsuperscript{MAD$\downarrow$}", "mad", True)
TASK_METRIC["athena_vsp"] = (r"\textsuperscript{MAD-norm}", "mad_norm", False)
TASK_METRIC["rms"]        = (r"\textsuperscript{F1}", "f1", False)

# Short rotated header per model, grouped by family
MODEL_DISPLAY = {
    "claude-sonnet-4-6-cyberxpert": "Claude-CX\\textsuperscript{*}",
    "GPT-5.4": "GPT-5.4\\textsuperscript{*}",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "Llama-Primus-Nemotron-70B-Instruct": "Primus-Nemo-70B",
    "Llama-Primus-Merged": "Primus-Merged",
    "Gemma-4-31B-it": "Gemma-4-31B",
    "Qwen3.6-35B-A3B": "Qwen3.6-35B",
    "Fanar-2-27B-Instruct": "Fanar-27B",
    "GPT-oss-20B": "GPT-oss-20B",
    "Foundation-Sec-8B-Instruct": "Found-Sec-8B",
    "RedSage-Qwen3-8B-DPO": "RedSage-8B",
}


def parse_cell(s: str):
    if s is None or s.strip() == "" or s.strip() == "NA":
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-task metric pulls — needed to match the original table's metrics.
# Cached so we don't re-read summary.json / detailed files per-cell.
# ─────────────────────────────────────────────────────────────────────────────


def _load_summary(model: str) -> dict:
    p = OUTPUTS_ROOT / f"judge_{model}" / "eval_results" / "summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _summary_entry(model: str, task: str) -> dict:
    s = _load_summary(model)
    return s.get("tasks", {}).get(task.upper(), {}) or {}


_F1_CACHE: dict = {}
_MAD_CACHE: dict = {}


def _compute_mad_from_detailed(model: str, task: str) -> float | None:
    """Mean MAD across per-sample CVSS vectors using the same scoring rule as
    `compute_vsp_metrics` (NONE / missing → 10.0 max penalty)."""
    key = (model, task)
    if key in _MAD_CACHE:
        return _MAD_CACHE[key]
    p = OUTPUTS_ROOT / f"judge_{model}" / "eval_results" / f"{task}_detailed.jsonl"
    if not p.exists():
        _MAD_CACHE[key] = None
        return None

    # Inline copy of calculate_vsp_mad from run_evaluate_llm_judge (so we don't
    # have to import torch via that module just to compute one CVSS distance).
    def _vsp_mad(pred: str, gold: str) -> float:
        try:
            from cvss import CVSS3
            def _norm(v: str) -> str:
                v = (v or "").strip()
                if v.startswith("CVSS:3.0/"):
                    return v.replace("CVSS:3.0/", "CVSS:3.1/")
                if v.startswith("CVSS:3.1/"):
                    return v
                return "CVSS:3.1/" + v
            return round(abs(CVSS3(_norm(pred)).scores()[0] - CVSS3(_norm(gold)).scores()[0]), 2)
        except Exception:
            return 10.0

    mads = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("skipped"):
                continue
            ext = r.get("extracted_answer", "") or ""
            if ext and ext.upper() != "NONE":
                mads.append(_vsp_mad(ext, r.get("ground_truth", "")))
            else:
                mads.append(10.0)
    res = (sum(mads) / len(mads)) if mads else None
    _MAD_CACHE[key] = res
    return res


def _compute_f1_from_detailed(model: str, task: str) -> float | None:
    """Compute macro-F1 over per-sample ID-set extractions.

    Reads detailed.jsonl for the (model, task) pair, parses ground_truth and
    extracted_answer as comma-separated ID sets (e.g. M1018,M1026 or
    T1027,T1059), and returns mean per-sample F1 across non-empty cells.
    Skipped rows excluded.
    """
    key = (model, task)
    if key in _F1_CACHE:
        return _F1_CACHE[key]
    p = OUTPUTS_ROOT / f"judge_{model}" / "eval_results" / f"{task}_detailed.jsonl"
    if not p.exists():
        _F1_CACHE[key] = None
        return None

    def _parse_ids(s: str) -> set:
        if s is None:
            return set()
        return {tok.strip().upper() for tok in re.split(r"[,;\s]+", str(s)) if tok.strip()}

    f1s = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("skipped"):
                continue
            gold = _parse_ids(r.get("ground_truth"))
            pred = _parse_ids(r.get("extracted_answer"))
            if not gold and not pred:
                continue
            if not pred:
                f1s.append(0.0)
                continue
            tp = len(gold & pred)
            fp = len(pred - gold)
            fn = len(gold - pred)
            if tp == 0:
                f1s.append(0.0)
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1s.append(2 * prec * rec / (prec + rec))
    res = (sum(f1s) / len(f1s)) if f1s else None
    _F1_CACHE[key] = res
    return res


def cell_value(model: str, task: str, kind: str,
               accuracy_fallback: float | None) -> float | None:
    """Return the metric value for (model, task) in its native units.

    `accuracy_fallback` is the strict-verdict accuracy already loaded from
    results_table.csv — used when kind == "acc" or as a sanity baseline.
    For "f1" / "mad" / "mad_norm" we go to summary.json or detailed.jsonl.
    Returned value is NOT scaled — formatting handles ×100 / decimal places.
    """
    if kind == "acc":
        return accuracy_fallback
    if kind == "mad":
        e = _summary_entry(model, task)
        v = e.get("mad") or e.get("mean_mad")
        if v is not None:
            return float(v)
        # summary.json was stripped (e.g. by retry_failed_judges) — recompute
        # from detailed.jsonl using the same rule as compute_vsp_metrics.
        return _compute_mad_from_detailed(model, task)
    if kind == "mad_norm":
        e = _summary_entry(model, task)
        v = e.get("mad") or e.get("mean_mad")
        if v is None:
            v = _compute_mad_from_detailed(model, task)
        if v is None:
            return None
        return max(0.0, 1.0 - float(v) / 7.7)
    if kind == "f1":
        e = _summary_entry(model, task)
        v = e.get("f1")
        if v is not None:
            return float(v)
        # RMS doesn't get f1 stamped by the judge — compute from detailed.jsonl.
        return _compute_f1_from_detailed(model, task)
    return accuracy_fallback


def fmt_value(v, kind: str):
    if v is None:
        return "--"
    if kind == "mad":
        return f"{v:.2f}"  # raw MAD, no percent
    return f"{v * 100:.1f}"


def fmt_value_bold(v, kind: str):
    if v is None:
        return "--"
    if kind == "mad":
        return f"\\textbf{{{v:.2f}}}"
    return f"\\textbf{{{v * 100:.1f}}}"


def main():
    rows = list(csv.reader(open(CSV_PATH)))
    header = rows[0]
    models_in_csv = header[1:]
    data = {}
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        task = r[0]
        data[task] = {m: parse_cell(r[1 + i]) for i, m in enumerate(models_in_csv)}

    # ── Resolve per-(task, model) cell value using the task's metric.
    # Replace `data` accuracy values with native-metric values where the metric
    # differs (MAD for vsp, MAD-norm for athena_vsp, F1 for rms).
    metric_data = {}
    for t in data:
        spec = TASK_METRIC.get(t, (r"\textsuperscript{Acc}", "acc", False))
        _, kind, _ = spec
        metric_data[t] = {}
        for m in models_in_csv:
            metric_data[t][m] = cell_value(m, t, kind, data[t].get(m))

    # Per-model averages — use accuracy values for the average (not mixed
    # metrics) so the headline Avg row stays a single comparable %.
    avgs = {}
    for m in models_in_csv:
        vals = [data[t][m] for t in data if data[t].get(m) is not None]
        avgs[m] = mean(vals) if vals else None

    # Sort: highest avg first; NAs last
    sorted_models = sorted(
        models_in_csv,
        key=lambda m: (avgs[m] is None, -(avgs[m] or 0.0)),
    )

    # Best per task across models — direction depends on lower_is_better.
    best_per_task = {}
    for t, row in metric_data.items():
        spec = TASK_METRIC.get(t, (r"\textsuperscript{Acc}", "acc", False))
        _, _, lower_is_better = spec
        finite = {m: v for m, v in row.items() if v is not None}
        if finite:
            best_per_task[t] = min(finite.values()) if lower_is_better else max(finite.values())

    # ── Build LaTeX ──────────────────────────────────────────────────────────
    out = []
    n_cols = len(sorted_models)
    col_spec = "@{}l" + "r" * n_cols + "@{}"

    out.append(r"\begin{table*}[t]")
    out.append(r"\centering")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(
        r"\caption{Master evaluation results across "
        f"{len(sorted_models)} models and {sum(len(g[1]) for g in PARENT_GROUPS)} sub-tasks "
        f"on {len(PARENT_GROUPS)} cybersecurity benchmark families "
        r"(CTI-Bench, AthenaBench, SECURE, RedSage-MCQ, CyberMetric, MMLU-CS, SecBench, SecEval, SEvenLLM). "
        r"\textbf{Bold}~=~best per task. Superscripts denote metric type: "
        r"\textsuperscript{Acc}~strict-verdict accuracy~(\%); "
        r"\textsuperscript{F1}~macro-F1 over the extracted ID set~(\%); "
        r"\textsuperscript{MAD$\downarrow$}~mean absolute deviation of CVSS base score "
        r"(lower is better; \textbf{bold}~=~lowest); "
        r"\textsuperscript{MAD-norm}~AthenaBench normalised MAD-accuracy "
        r"$=\max(0,1-\text{MAD}/7.7)\times 100$~(\%). All cells reflect per-sample "
        r"majority vote of up to three GPT-5.4 judge runs (default, v1, v2) under the "
        r"unified extract-and-verdict prompt of Section~\ref{sec:protocol}. "
        r"\textsuperscript{*}~closed/API model. Models are ordered by per-model "
        r"strict-verdict average (\textsuperscript{Acc} across all populated cells); "
        r"`--' indicates a cell has no judged samples.}"
    )
    out.append(r"\label{tab:main_results}")
    out.append(r"\resizebox{\linewidth}{!}{%")
    out.append(rf"\begin{{tabular}}{{{col_spec}}}")
    out.append(r"\toprule")

    # Header row with rotated model names
    h = [r"\textbf{Task}"]
    for m in sorted_models:
        h.append(rf"\rotatebox{{65}}{{\textbf{{{MODEL_DISPLAY.get(m, m)}}}}}")
    out.append(" & ".join(h) + r" \\")
    out.append(r"\midrule")

    # Per-parent groups
    for parent, tasks in PARENT_GROUPS:
        out.append(rf"\multicolumn{{{n_cols + 1}}}{{l}}{{\textit{{{parent}}}}}\\")
        for t in tasks:
            if t not in metric_data:
                continue
            row = metric_data[t]
            spec = TASK_METRIC.get(t, (r"\textsuperscript{Acc}", "acc", False))
            sup, kind, _ = spec
            label = TASK_DISPLAY.get(t, t) + " " + sup
            cells = [label]
            best = best_per_task.get(t)
            for m in sorted_models:
                v = row.get(m)
                if v is not None and best is not None and abs(v - best) < 1e-9:
                    cells.append(fmt_value_bold(v, kind))
                else:
                    cells.append(fmt_value(v, kind))
            out.append(" & ".join(cells) + r" \\")
        out.append(r"\midrule")

    # Average row — uses strict-verdict accuracy across all populated cells
    # (not the per-task native metric) so it stays comparable across columns.
    avg_cells = [r"\textit{Average} \textsuperscript{Acc}"]
    finite_avgs = [avgs[m] for m in sorted_models if avgs[m] is not None]
    best_avg = max(finite_avgs) if finite_avgs else None
    for m in sorted_models:
        a = avgs[m]
        if a is not None and best_avg is not None and abs(a - best_avg) < 1e-9:
            avg_cells.append(rf"\textbf{{{a * 100:.1f}}}")
        else:
            avg_cells.append(fmt_value(a, "acc"))
    out.append(" & ".join(avg_cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"}% resizebox")
    out.append(r"\end{table*}")

    print("\n".join(out))


if __name__ == "__main__":
    main()
