"""Build the deduped flagged-sample bank for verification.

Reads the per-task flagged JSONLs and the responses files, then writes a
single JSONL `analysis/reports/verification/flagged_bank.jsonl` keyed by
`(task, index)` containing:

  - task, index
  - gold, majority_prediction
  - agreement_fraction, n_models_with_data, agreeing_models
  - all_predictions (model -> normalised prediction)
  - first_threshold_at_or_above (the highest threshold at which this sample is flagged)
  - question (full prompt text from responses file, NOT truncated)

Sorted by descending first_threshold so verification can walk top-down.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis.lib.loaders import iter_jsonl, responses_path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
REPORTS = HERE / "reports"
OUT_DIR = REPORTS / "verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds we care about, descending. The bank includes every sample
# whose agreement_fraction >= MIN_THRESHOLD.
# 0.50 added to support the broader (search-only) audit; the τ≥0.75 entries
# are unchanged and their existing verdicts can be reused.
THRESHOLDS = [1.00, 0.90, 5/6, 0.75, 0.50]   # (5/6 == 0.8333…)
MIN_THRESHOLD = 0.50

FLAGGED_DIR = REPORTS / "gold_errors" / "flagged"


def first_threshold_at_or_above(frac: float) -> float:
    for t in THRESHOLDS:
        if frac >= t - 1e-9:
            return round(t, 4)
    return 0.0


_QUESTION_CACHE: dict = {}


def load_questions_for_task(task: str) -> dict:
    """Return a dict of idx -> full prompt text for one task, cached."""
    if task in _QUESTION_CACHE:
        return _QUESTION_CACHE[task]
    by_idx: dict = {}
    for child in OUTPUTS_ROOT.iterdir():
        if not child.name.startswith("responses_"):
            continue
        m = child.name[len("responses_"):]
        path = responses_path(OUTPUTS_ROOT, m, task)
        if not path.exists():
            continue
        for sample in iter_jsonl(path):
            idx = str(sample.get("index", ""))
            if idx in by_idx:
                continue
            q = sample.get("prompt", "")
            if isinstance(q, list):
                q = "\n".join(
                    msg.get("content", "") for msg in q
                    if isinstance(msg, dict)
                )
            if q:
                by_idx[idx] = q
        if len(by_idx) > 0:
            # one model already covered all indices for this task
            break
    _QUESTION_CACHE[task] = by_idx
    return by_idx


def lookup_full_question(task: str, idx: str, models_already_in_record: list) -> str:
    by_idx = load_questions_for_task(task)
    return by_idx.get(str(idx), "")


def main():
    bank = []
    for path in sorted(FLAGGED_DIR.glob("*_t50.jsonl")):
        task = path.stem.replace("_t50", "")
        for rec in iter_jsonl(path):
            frac = rec.get("agreement_fraction", 0.0)
            if frac < MIN_THRESHOLD - 1e-9:
                continue
            tier = first_threshold_at_or_above(frac)
            agreeing = rec.get("agreeing_models", []) or []
            full_q = lookup_full_question(task, rec["index"], agreeing)
            bank.append({
                "task": task,
                "index": rec["index"],
                "gold": rec["gold"],
                "majority_prediction": rec["majority_prediction"],
                "agreement_fraction": frac,
                "n_models_with_data": rec.get("n_models_with_data"),
                "agreeing_models": agreeing,
                "all_predictions": rec.get("all_predictions", {}),
                "first_threshold_at_or_above": tier,
                "question": full_q,
            })

    bank.sort(key=lambda r: (-r["first_threshold_at_or_above"], r["task"], int(r["index"])))

    out = OUT_DIR / "flagged_bank.jsonl"
    with open(out, "w") as f:
        for r in bank:
            json.dump(r, f)
            f.write("\n")

    summary_path = OUT_DIR / "flagged_bank_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "n_samples_at_or_above"])
        cum = {t: 0 for t in THRESHOLDS}
        for r in bank:
            for t in THRESHOLDS:
                if r["agreement_fraction"] >= t - 1e-9:
                    cum[t] += 1
        for t in THRESHOLDS:
            w.writerow([f"{t:.4f}", cum[t]])

    print(f"flagged bank: {len(bank)} samples → {out}")
    for t in THRESHOLDS:
        print(f"  >= {t:.4f}: {cum[t]} samples")


if __name__ == "__main__":
    main()
