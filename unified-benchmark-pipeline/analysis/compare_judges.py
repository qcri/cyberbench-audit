#!/usr/bin/env python3
"""Compare LLM-as-a-judge verdicts across multiple judges.

Joins per-task `*_detailed.jsonl` files from different judge runs by `index`
and surfaces samples where the judges disagree (verdict and/or extracted_answer).

Layout assumed:
    outputs/judge_<MODEL>/eval_results/<task>_detailed.jsonl                   (legacy: GPT-5.4)
    outputs/judge_<MODEL>__by_<JUDGE_ALIAS>/eval_results/<task>_detailed.jsonl (new)

Usage:
    python compare_judges.py --model GPT-5.4 \
        --judges gpt-5.4 claude-sonnet-4-6 qwen3.6-35b-a3b \
        --tasks mcq rcm vsp \
        --out outputs/judge_compare/GPT-5.4

The first judge alias is treated as the "anchor" — its rows define the row set.
Output:
    <out>/<task>_disagreements.jsonl   one row per sample where any pair of
                                       judges disagree on `is_correct`
    <out>/<task>_summary.json          per-task agreement counts
    <out>/summary.json                 per-judge & cross-judge totals
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List


def judge_dir(outputs_root: Path, model: str, alias: str) -> Path:
    """Return the eval_results dir for (model, judge alias).

    The legacy GPT-5.4 results live in `judge_<MODEL>/eval_results` without a
    judge suffix; new judges live in `judge_<MODEL>__by_<ALIAS>/eval_results`.
    """
    legacy = outputs_root / f"judge_{model}" / "eval_results"
    suffixed = outputs_root / f"judge_{model}__by_{alias}" / "eval_results"
    if suffixed.exists():
        return suffixed
    if alias.lower() in {"gpt-5.4", "gpt5.4", "gpt_5_4"} and legacy.exists():
        return legacy
    return suffixed  # may not exist; caller handles


def load_task(path: Path) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                rows[int(obj["index"])] = obj
            except (KeyError, ValueError):
                continue
    return rows


def compare_task(model: str, task: str, judges: List[str],
                 outputs_root: Path) -> tuple[List[dict], dict]:
    """Return (disagreement_rows, summary)."""
    per_judge: Dict[str, Dict[int, dict]] = {}
    for j in judges:
        d = judge_dir(outputs_root, model, j)
        per_judge[j] = load_task(d / f"{task}_detailed.jsonl")

    sizes = {j: len(per_judge[j]) for j in judges}
    if not any(sizes.values()):
        return [], {"task": task, "sizes": sizes, "note": "no rows for any judge"}

    # Use the union of indices but only consider samples present in all judges
    # for the disagreement analysis (otherwise we'd just be measuring missing files).
    common = set.intersection(*(set(p.keys()) for p in per_judge.values())) if all(sizes.values()) else set()

    disagreements = []
    pair_disagreements = Counter()
    verdict_dist = defaultdict(Counter)
    extracted_disagreements = 0

    for idx in sorted(common):
        rows = {j: per_judge[j][idx] for j in judges}
        verdicts = {j: bool(rows[j].get("is_correct")) for j in judges}
        extracted = {j: (rows[j].get("extracted_answer") or "").strip() for j in judges}

        for j, v in verdicts.items():
            verdict_dist[j][str(v)] += 1

        verdict_disagree = len(set(verdicts.values())) > 1
        extracted_disagree = len(set(e for e in extracted.values())) > 1

        if verdict_disagree:
            for a, b in combinations(judges, 2):
                if verdicts[a] != verdicts[b]:
                    pair_disagreements[f"{a}__vs__{b}"] += 1
        if extracted_disagree:
            extracted_disagreements += 1

        if verdict_disagree or extracted_disagree:
            row = {
                "index": idx,
                "task": task,
                "question": rows[judges[0]].get("question", ""),
                "ground_truth": rows[judges[0]].get("ground_truth", ""),
                "model_response": rows[judges[0]].get("model_response", ""),
                "verdict_disagreement": verdict_disagree,
                "extracted_disagreement": extracted_disagree,
                "by_judge": {
                    j: {
                        "is_correct": verdicts[j],
                        "extracted_answer": extracted[j],
                        "judge_justification": rows[j].get("judge_justification", ""),
                    }
                    for j in judges
                },
            }
            disagreements.append(row)

    summary = {
        "task": task,
        "model": model,
        "judges": judges,
        "row_counts": sizes,
        "common_rows": len(common),
        "verdict_disagreements": sum(1 for d in disagreements if d["verdict_disagreement"]),
        "extracted_disagreements": extracted_disagreements,
        "pair_verdict_disagreements": dict(pair_disagreements),
        "verdict_distribution": {j: dict(c) for j, c in verdict_dist.items()},
    }
    return disagreements, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Evaluated model name, e.g. GPT-5.4")
    ap.add_argument("--judges", nargs="+", required=True,
                    help="Judge aliases, e.g. gpt-5.4 claude-sonnet-4-6 qwen3.6-35b-a3b")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Tasks to compare. Default: all tasks present for the first judge.")
    ap.add_argument("--outputs_root", default=None,
                    help="Path to outputs/ dir. Default: <repo>/outputs.")
    ap.add_argument("--out", required=True, help="Directory to write comparison results into.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    outputs_root = Path(args.outputs_root) if args.outputs_root else (repo_root / "outputs")

    if args.tasks is None:
        anchor_dir = judge_dir(outputs_root, args.model, args.judges[0])
        if not anchor_dir.exists():
            raise SystemExit(f"Anchor judge dir not found: {anchor_dir}")
        args.tasks = sorted({p.name.replace("_detailed.jsonl", "")
                             for p in anchor_dir.glob("*_detailed.jsonl")})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = {
        "model": args.model,
        "judges": args.judges,
        "outputs_root": str(outputs_root),
        "tasks": {},
        "totals": {
            "common_rows": 0,
            "verdict_disagreements": 0,
            "extracted_disagreements": 0,
            "pair_verdict_disagreements": Counter(),
        },
    }

    for task in args.tasks:
        disagreements, summary = compare_task(args.model, task, args.judges, outputs_root)
        with (out_dir / f"{task}_disagreements.jsonl").open("w") as f:
            for r in disagreements:
                f.write(json.dumps(r) + "\n")
        with (out_dir / f"{task}_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        overall["tasks"][task] = summary
        overall["totals"]["common_rows"] += summary.get("common_rows", 0)
        overall["totals"]["verdict_disagreements"] += summary.get("verdict_disagreements", 0)
        overall["totals"]["extracted_disagreements"] += summary.get("extracted_disagreements", 0)
        for k, v in summary.get("pair_verdict_disagreements", {}).items():
            overall["totals"]["pair_verdict_disagreements"][k] += v

        print(f"[{task}] common={summary.get('common_rows',0)} "
              f"verdict_disagree={summary.get('verdict_disagreements',0)} "
              f"extracted_disagree={summary.get('extracted_disagreements',0)}")

    overall["totals"]["pair_verdict_disagreements"] = dict(overall["totals"]["pair_verdict_disagreements"])
    with (out_dir / "summary.json").open("w") as f:
        json.dump(overall, f, indent=2)

    print(f"\nWrote comparison to {out_dir}/")
    print(f"Total verdict disagreements: {overall['totals']['verdict_disagreements']} "
          f"over {overall['totals']['common_rows']} common rows")


if __name__ == "__main__":
    main()
