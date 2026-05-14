"""Companion table: F1 / MAD side-metrics for ATE-style and VSP-style tasks.

The strict-verdict accuracy in `results_table.csv` undercounts model
correctness on free-form ID-extraction (ATE/ATHENA_ATE) and CVSS-vector
scoring (VSP/ATHENA_VSP). The default judge's summary.json carries the
traditional benchmark metrics (F1 for IDs, mean MAD for CVSS) alongside the
strict accuracy. This script writes:

  reports/secondary_metrics.csv
  reports/secondary_metrics.md

so canonical reporting can use F1 / MAD when that is the appropriate metric.
"""

from __future__ import annotations

import csv
from pathlib import Path

from analysis.lib.loaders import discover_models, load_summary_entry


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
REPORTS = HERE / "reports"

ID_TASKS = ["ate", "athena_ate"]
VSP_TASKS = ["vsp", "athena_vsp"]


def fmt(v):
    if v is None:
        return "NA"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    models = discover_models(OUTPUTS_ROOT)
    rows = []
    for task in ID_TASKS:
        for m in models:
            entry = load_summary_entry(OUTPUTS_ROOT, m, task, v1=False) or {}
            rows.append({
                "task": task,
                "model": m,
                "kind": "id_set",
                "strict_accuracy": fmt(entry.get("accuracy")),
                "exact_matches": entry.get("exact_matches", ""),
                "precision": fmt(entry.get("precision")),
                "recall": fmt(entry.get("recall")),
                "f1": fmt(entry.get("f1")),
                "tp_total": entry.get("tp_total", ""),
                "fp_total": entry.get("fp_total", ""),
                "fn_total": entry.get("fn_total", ""),
                "mad": "",
                "extraction_success": "",
            })
    for task in VSP_TASKS:
        for m in models:
            entry = load_summary_entry(OUTPUTS_ROOT, m, task, v1=False) or {}
            rows.append({
                "task": task,
                "model": m,
                "kind": "cvss_vector",
                "strict_accuracy": fmt(entry.get("accuracy")),
                "exact_matches": entry.get("correct", ""),
                "precision": "",
                "recall": "",
                "f1": "",
                "tp_total": "",
                "fp_total": "",
                "fn_total": "",
                "mad": fmt(entry.get("mad") or entry.get("mean_mad")),
                "extraction_success": entry.get("extraction_success", ""),
            })

    fieldnames = list(rows[0].keys())
    with open(REPORTS / "secondary_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Markdown view per task
    md = ["# Secondary metrics (canonical reporting)",
          "",
          "Strict-verdict accuracy in the master table undercounts model performance on"
          " ID-extraction tasks (ATE / ATHENA_ATE) and CVSS-vector tasks (VSP / ATHENA_VSP)."
          " For canonical comparison, use the F1 / MAD columns below.",
          ""]
    for task in ID_TASKS:
        md.append(f"## {task} — F1 (set overlap)")
        md.append("")
        md.append("| Model | strict acc. | exact matches | precision | recall | F1 |")
        md.append("|---|---|---|---|---|---|")
        for r in [x for x in rows if x["task"] == task]:
            md.append(
                f"| {r['model']} | {r['strict_accuracy']} | {r['exact_matches']} | "
                f"{r['precision']} | {r['recall']} | {r['f1']} |"
            )
        md.append("")
    for task in VSP_TASKS:
        md.append(f"## {task} — Mean Absolute Deviation (lower is better)")
        md.append("")
        md.append("| Model | strict acc. | exact matches | mean MAD | extracted |")
        md.append("|---|---|---|---|---|")
        for r in [x for x in rows if x["task"] == task]:
            md.append(
                f"| {r['model']} | {r['strict_accuracy']} | {r['exact_matches']} | "
                f"{r['mad']} | {r['extraction_success']} |"
            )
        md.append("")
    (REPORTS / "secondary_metrics.md").write_text("\n".join(md))

    print(f"wrote {REPORTS}/secondary_metrics.{{csv,md}}")


if __name__ == "__main__":
    main()
