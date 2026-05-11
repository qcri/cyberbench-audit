"""Build the stratified K/A classification sample bank.

For each of 24 sub-tasks, sample 100 questions uniformly at random (seed=0)
from the full responses file (any one model — prompts are model-invariant).
Save to analysis/reports/coverage/sample_bank.jsonl with fields
{task, index, question}, where `question` is the boilerplate-stripped prompt.

We use the same `strip_boilerplate` from the embedding pipeline so the
classifier sees the same content the embedder did.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from analysis.lib.embeddings import iter_task_questions
from analysis.lib.loaders import TASK_ORDER


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
REPORTS = HERE / "reports"
OUT_DIR = REPORTS / "coverage"

SAMPLES_PER_TASK = 100
SEED = 0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    bank = []
    for task in TASK_ORDER:
        rows = list(iter_task_questions(OUTPUTS_ROOT, task))
        if not rows:
            print(f"  [{task}] no rows; skipping")
            continue
        if len(rows) <= SAMPLES_PER_TASK:
            picked = rows
        else:
            picked = rng.sample(rows, SAMPLES_PER_TASK)
        for idx, text in picked:
            bank.append({
                "task": task,
                "index": idx,
                "question": text,
            })
        print(f"  [{task}] picked {len(picked)} / {len(rows)}")

    out = OUT_DIR / "sample_bank.jsonl"
    with open(out, "w") as f:
        for r in bank:
            json.dump(r, f); f.write("\n")
    print(f"\nwrote {len(bank)} samples -> {out}")


if __name__ == "__main__":
    main()
