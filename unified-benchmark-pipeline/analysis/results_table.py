"""Step 1 — Master accuracy table (majority vote across judges).

For every (model, sub-task) cell, we collect per-sample `is_correct` flags from
every available judge run — default (`judge_<MODEL>/`), v1 (`judge_<MODEL>_v1/`),
and v2 (`judge_<MODEL>_v2/`) — align them by sample `index`, and take the
majority verdict per index. Cell accuracy is then the mean of the per-index
majority verdicts.

Note on alignment: default and v1 judge the same `responses_<MODEL>/` file, so
their per-index flags align. v2 judges `responses_<MODEL>_v2/`, a different
inference run, but indices typically still line up for the 22 shared tasks
(both pipelines emit a 0..N-1 index over the same dataset). When v2 indexes
that the other judges don't have, we still include them — every sample with
at least one verdict contributes.

Outputs (under analysis/reports/):
  - results_table.csv          numeric accuracies, NA where missing
  - results_table.md           grouped pretty view with parent-benchmark headers
  - results_table_meta.csv     per-cell source label + sample count
  - per_model_task_accuracy.json  cached map for downstream scripts

Run from BenchmarkingSecBenchmarks/:
  python -m analysis.results_table
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

from analysis.lib.loaders import (
    JUDGE_VERSIONS,
    PARENT_GROUPS,
    TASK_ORDER,
    detailed_path,
    discover_models,
    iter_jsonl,
    load_summary_accuracy,
    load_summary_n,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
REPORTS_DIR = HERE / "reports"


# Short tag per judge version for the meta CSV's source column.
_VERSION_TAG = {"": "def", "_v1": "v1", "_v2": "v2"}


def _load_per_index_correct(model: str, task: str, version: str) -> Dict[str, bool]:
    """Load {sample_index_str -> is_correct_bool} for one judge version.

    Skipped samples (judge API failures) are excluded — they shouldn't count
    as a vote in either direction.
    """
    path = detailed_path(OUTPUTS_ROOT, model, task, version=version)
    if not path.exists():
        return {}
    out: Dict[str, bool] = {}
    for r in iter_jsonl(path):
        if "is_correct" not in r:
            continue
        if r.get("skipped"):
            continue
        idx = str(r.get("index", ""))
        if idx == "":
            continue
        out[idx] = bool(r["is_correct"])
    return out


def cell_accuracy(model: str, task: str) -> Tuple[Optional[float], str, int]:
    """Per-cell accuracy via per-sample majority vote across judges.

    Returns (accuracy, source_label, n_samples).
      - source_label: which judges contributed, e.g. "def+v1+v2", "def+v1",
        "v2", "missing".
      - n_samples: number of indices with at least one verdict.
    """
    per_judge = {ver: _load_per_index_correct(model, task, ver)
                 for ver in JUDGE_VERSIONS}

    # union of all sample indices that any judge scored
    all_idx = set()
    for d in per_judge.values():
        all_idx.update(d.keys())

    if not all_idx:
        return None, "missing", 0

    correct = 0
    for idx in all_idx:
        votes = [per_judge[v][idx] for v in JUDGE_VERSIONS if idx in per_judge[v]]
        if not votes:
            continue
        # Strict majority: more than half of contributing judges must say CORRECT.
        # Ties (e.g., 1 correct / 1 incorrect with 2 judges) are not counted as
        # correct — tied evidence should not inflate accuracy.
        if sum(votes) * 2 > len(votes):
            correct += 1
    n = len(all_idx)

    contributors = [_VERSION_TAG[v] for v in JUDGE_VERSIONS if per_judge[v]]
    source = "+".join(contributors) if contributors else "missing"
    return correct / n, source, n


def _legacy_detailed_for(model: str, task: str, version: str = ""):
    """Back-compat helper kept for external callers."""
    return detailed_path(OUTPUTS_ROOT, model, task, version=version)


def iter_records(path):
    return iter_jsonl(path)


def fmt(acc):
    return "NA" if acc is None else f"{acc:.4f}"


def write_csv(path: Path, models: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", *models])
        for row in rows:
            writer.writerow([row["task"], *(fmt(row["acc"][m]) for m in models)])


def write_meta_csv(path: Path, models: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["task"]
        for m in models:
            header.extend([f"{m}_source", f"{m}_n"])
        writer.writerow(header)
        for row in rows:
            cells = [row["task"]]
            for m in models:
                cells.extend([row["source"][m], str(row["n"][m])])
            writer.writerow(cells)


def write_md(path: Path, models: List[str], rows_by_task: Dict[str, dict]) -> None:
    lines = []
    lines.append("# Master accuracy table")
    lines.append("")
    lines.append("Cell value = strict-verdict accuracy from the default judge "
                 "(summary.json preferred; otherwise mean of per-sample is_correct). "
                 "`NA` means the default judge has no result for that (model, task) pair.")
    lines.append("")

    header = "| Sub-task | " + " | ".join(models) + " |"
    sep = "|" + "|".join(["---"] * (len(models) + 1)) + "|"

    per_model_avgs = {m: [] for m in models}  # type: Dict[str, List[float]]

    for parent, tasks in PARENT_GROUPS:
        lines.append(f"## {parent}")
        lines.append("")
        lines.append(header)
        lines.append(sep)

        parent_acc = {m: [] for m in models}  # type: Dict[str, List[float]]
        for task in tasks:
            row = rows_by_task.get(task)
            if row is None:
                continue
            cells = []
            for m in models:
                a = row["acc"][m]
                cells.append(fmt(a))
                if a is not None:
                    parent_acc[m].append(a)
                    per_model_avgs[m].append(a)
            lines.append(f"| {task} | " + " | ".join(cells) + " |")

        macro = []
        for m in models:
            macro.append(fmt(mean(parent_acc[m])) if parent_acc[m] else "NA")
        lines.append(f"| **{parent} macro-avg** | " + " | ".join(macro) + " |")
        lines.append("")

    lines.append("## Overall (per-model average across all populated cells)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    overall = []
    for m in models:
        overall.append(fmt(mean(per_model_avgs[m])) if per_model_avgs[m] else "NA")
    lines.append(f"| Overall | " + " | ".join(overall) + " |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    models = discover_models(OUTPUTS_ROOT)

    rows = []
    rows_by_task = {}  # type: Dict[str, dict]
    cache = {m: {} for m in models}  # type: Dict[str, Dict[str, Optional[float]]]

    for task in TASK_ORDER:
        row = {"task": task, "acc": {}, "source": {}, "n": {}}
        for m in models:
            acc, source, n = cell_accuracy(m, task)
            row["acc"][m] = acc
            row["source"][m] = source
            row["n"][m] = n
            cache[m][task] = acc
        rows.append(row)
        rows_by_task[task] = row

    write_csv(REPORTS_DIR / "results_table.csv", models, rows)
    write_meta_csv(REPORTS_DIR / "results_table_meta.csv", models, rows)
    write_md(REPORTS_DIR / "results_table.md", models, rows_by_task)

    cache_path = REPORTS_DIR / "per_model_task_accuracy.json"
    cache_path.write_text(json.dumps(cache, indent=2))

    n_models = len(models)
    n_filled = sum(
        1
        for row in rows
        for m in models
        if row["acc"][m] is not None
    )
    print(
        f"results_table: {n_models} models x {len(rows)} tasks "
        f"= {n_models * len(rows)} cells, {n_filled} populated"
    )
    print(f"Wrote {REPORTS_DIR}/{{results_table.csv,results_table.md,results_table_meta.csv,per_model_task_accuracy.json}}")


if __name__ == "__main__":
    main()
