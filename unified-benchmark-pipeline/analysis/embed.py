"""Embed every benchmark question with sentence-transformers, cache per task.

Outputs: analysis/reports/embeddings/embeddings/<task>.npz
  containing: indices  (str array, n)
              vectors  (float32 array, n x 768)

Resumable: skips a task if the cache size already matches the source size.

Run via slurm/run_embed.sh (GPU partition). Single-call CLI:

    PYTHONPATH=. <conda-python> -m analysis.embed [--model NAME] [--task NAME]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.lib.embeddings import encode_batch, iter_task_questions
from analysis.lib.loaders import TASK_ORDER


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
REPORTS = HERE / "reports"
EMBED_DIR = REPORTS / "embeddings" / "embeddings"


def cache_path(task: str) -> Path:
    return EMBED_DIR / f"{task}.npz"


def existing_size(task: str) -> int:
    path = cache_path(task)
    if not path.exists():
        return 0
    try:
        with np.load(path, allow_pickle=False) as z:
            return int(z["vectors"].shape[0])
    except Exception:
        return 0


def expected_size(task: str) -> int:
    return sum(1 for _ in iter_task_questions(OUTPUTS_ROOT, task))


def embed_task(task: str, model_name: str, batch_size: int = 64) -> None:
    rows = list(iter_task_questions(OUTPUTS_ROOT, task))
    if not rows:
        print(f"  [{task}] no input rows; skipping")
        return
    have = existing_size(task)
    if have == len(rows):
        print(f"  [{task}] cache up to date ({have} samples); skip")
        return

    indices = [idx for idx, _ in rows]
    texts = [text for _, text in rows]
    print(f"  [{task}] encoding {len(texts)} samples...", flush=True)
    vectors = encode_batch(texts, model_name=model_name, batch_size=batch_size)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path(task),
        indices=np.asarray(indices, dtype=object),
        vectors=vectors.astype(np.float32),
    )
    print(f"  [{task}] wrote {cache_path(task)} shape={vectors.shape}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--task", default=None,
                       help=f"Embed only this task (default: all {len(TASK_ORDER)}).")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [args.task] if args.task else TASK_ORDER
    print(f"model: {args.model}")
    print(f"tasks: {len(tasks)}")
    for t in tasks:
        try:
            embed_task(t, args.model, args.batch_size)
        except Exception as e:
            print(f"  [{t}] ERROR: {type(e).__name__}: {e}", flush=True)
            raise


if __name__ == "__main__":
    main()
