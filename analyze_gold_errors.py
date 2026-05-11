#!/usr/bin/env python3
"""
Majority-voting analysis to flag potentially wrong gold answers in benchmark datasets.

For each sample in each task, collect predictions from N models. If ≥ threshold% of
models agree on an answer different from the gold label, flag the sample as potentially
having an incorrect gold answer.

Usage:
    python analyze_gold_errors.py [--output_dir results/gold_error_analysis]
"""

import json
import re
import os
import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

# ── Models to include ─────────────────────────────────────────────────────────
MODELS = [
    "Llama-Primus-Merged",
    "Foundation-Sec-8B-Instruct",
    "RedSage-Qwen3-8B-DPO",
    "Qwen3.6-35B-A3B",
    "GPT-oss-20B",
    "Fanar-2-27B-Instruct",
]

# ── Thresholds (fraction of models that must agree) ───────────────────────────
THRESHOLDS = [2/6, 3/6, 4/6, 5/6, 6/6]  # 33%, 50%, 67%, 83%, 100%
THRESHOLD_LABELS = ["2/6 (33%)", "3/6 (50%)", "4/6 (67%)", "5/6 (83%)", "6/6 (100%)"]

# ── Task types ────────────────────────────────────────────────────────────────
# Tasks where prediction is a letter or set of letters (A-D)
MCQ_TASKS = {
    "mcq", "cybermetric", "mmlu_cs", "mmlu-cs", "seceval", "secbench",
    "secure_maet", "secure_cwet",
    "ckt", "redsage_frameworks", "redsage_generals", "redsage_kali",
    "redsage_cli", "redsage_skills",
}

# True/False/X — secure_kcv uses single-letter T/F/X labels, not A-D.
TFX_TASKS = {"secure_kcv"}

# Tasks where prediction is extracted IDs (T#### or M####)
ID_TASKS = {"ate", "athena_ate", "rms"}

# Tasks where prediction is a free-form answer (harder to vote on directly)
TEXT_TASKS = {"rcm", "athena_rcm", "taa"}

# VSP: CVSS vector string
VSP_TASKS = {"vsp", "athena_vsp"}


def strip_think(text: str) -> str:
    """Strip CoT thinking chain from reasoning models (e.g. Qwen3)."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def extract_mcq_answer(response: str) -> str:
    """Extract letter answer(s) from MCQ response. Returns sorted string like 'AB' or 'B'."""
    response = strip_think(response)
    letters = sorted(set(re.findall(r'\b([A-Da-d])\b', response)))
    return "".join(l.upper() for l in letters) if letters else ""


def extract_ids(text: str, pattern: str) -> frozenset:
    """Extract technique/mitigation IDs from text."""
    matches = re.findall(pattern, text, re.IGNORECASE)
    return frozenset(m.upper() for m in matches)


def extract_technique_ids(text: str) -> frozenset:
    """Extract MITRE ATT&CK technique IDs (T####[.###])."""
    return extract_ids(text, r'T\d{4}(?:\.\d{3})?')


def extract_mitigation_ids(text: str) -> frozenset:
    """Extract MITRE mitigation IDs (M####)."""
    return extract_ids(text, r'M\d{4}')


def extract_tfx_answer(response: str) -> str:
    """Extract T/F/X label from SECURE-KCV response. Returns last T, F, or X found."""
    response = strip_think(response)
    matches = re.findall(r'\b([TFXtfx])\b', response)
    return matches[-1].upper() if matches else ""


def extract_cvss_vector(text: str) -> str:
    """Extract CVSS v3 vector string."""
    text = strip_think(text)
    pattern = r'(?:CVSS:3\.[01]/)?AV:[A-Z]+/AC:[A-Z]+/PR:[A-Z]+/UI:[A-Z]+/S:[A-Z]+/C:[A-Z]+/I:[A-Z]+/A:[A-Z]+'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0).upper() if match else ""


def normalize_gold(gold: str, task: str) -> str:
    """Normalize gold answer to same format as extracted prediction."""
    task = task.replace("-", "_").lower()
    if task in MCQ_TASKS:
        letters = sorted(set(re.findall(r'[A-Da-d]', gold)))
        return "".join(l.upper() for l in letters)
    if task in TFX_TASKS:
        g = gold.strip().upper()
        return g[0] if g and g[0] in {"T", "F", "X"} else ""
    return gold.strip()


def extract_prediction(response: str, task: str) -> Optional[str]:
    """Extract normalized prediction from model response based on task type."""
    task = task.replace("-", "_").lower()
    response = response.strip()
    if not response:
        return None

    if task in MCQ_TASKS:
        pred = extract_mcq_answer(response)
        return pred if pred else None

    elif task in TFX_TASKS:
        pred = extract_tfx_answer(response)
        return pred if pred else None

    elif task in ID_TASKS:
        if "ate" in task:
            ids = extract_technique_ids(strip_think(response))
        else:
            ids = extract_mitigation_ids(strip_think(response))
        return str(sorted(ids)) if ids else None

    elif task in VSP_TASKS:
        vec = extract_cvss_vector(response)
        return vec if vec else None

    elif task in TEXT_TASKS:
        # For free-text tasks, use the full stripped response (normalized)
        text = strip_think(response).lower().strip()
        return text[:200] if text else None

    return None


def load_responses(response_dir: str, task: str) -> dict:
    """Load responses for a task, keyed by sample index."""
    # Try both underscore and hyphen variants
    for fname in [f"{task}_responses.jsonl", f"{task.replace('_', '-')}_responses.jsonl"]:
        path = Path(response_dir) / fname
        if path.exists():
            samples = {}
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        idx = str(r.get("index", ""))
                        samples[idx] = r
                    except json.JSONDecodeError:
                        continue
            return samples
    return {}


def analyze_task(task: str, responses_root: str, models: list) -> dict:
    """
    For a single task, collect predictions from all models and apply majority voting.

    Returns dict with:
        - total_samples: int
        - n_models: int (models that had data for this task)
        - flagged: dict[threshold_label -> list of flagged sample dicts]
    """
    # Load all model responses for this task
    model_data = {}
    for model in models:
        response_dir = Path(responses_root) / f"responses_{model}"
        samples = load_responses(str(response_dir), task)
        if samples:
            model_data[model] = samples

    if not model_data:
        return None

    # Collect all sample indices
    all_indices = set()
    for samples in model_data.values():
        all_indices.update(samples.keys())

    n_models = len(model_data)
    flagged_by_threshold = {label: [] for label in THRESHOLD_LABELS}

    for idx in sorted(all_indices, key=lambda x: int(x) if x.isdigit() else x):
        # Collect predictions from each model
        predictions = {}
        gold = None
        question = None

        for model, samples in model_data.items():
            sample = samples.get(idx)
            if sample is None:
                continue
            if gold is None:
                gold = normalize_gold(sample.get("ground_truth", ""), task)
                question = sample.get("prompt", "")
                if isinstance(question, list):
                    # Few-shot format: grab last user message
                    for msg in reversed(question):
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            question = msg.get("content", "")
                            break
                    else:
                        question = str(question)

            resp = sample.get("model_response", "")
            pred = extract_prediction(resp, task)
            if pred is not None:
                predictions[model] = pred

        if not predictions or gold is None:
            continue

        # Majority vote
        pred_counts = Counter(predictions.values())
        most_common_pred, most_common_count = pred_counts.most_common(1)[0]

        # Flag if majority prediction differs from gold
        if most_common_pred != gold:
            agreement_frac = most_common_count / len(predictions)
            agreeing_models = [m for m, p in predictions.items() if p == most_common_pred]
            all_preds = dict(predictions)

            record = {
                "index": idx,
                "gold": gold,
                "majority_prediction": most_common_pred,
                "agreement_count": most_common_count,
                "agreement_fraction": round(agreement_frac, 3),
                "n_models_with_data": len(predictions),
                "agreeing_models": agreeing_models,
                "all_predictions": all_preds,
                "question_snippet": str(question)[:300],
            }

            for thresh, label in zip(THRESHOLDS, THRESHOLD_LABELS):
                if agreement_frac >= thresh:
                    flagged_by_threshold[label].append(record)

    return {
        "total_samples": len(all_indices),
        "n_models": n_models,
        "flagged": flagged_by_threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Flag potentially wrong gold answers via majority voting")
    parser.add_argument("--responses_root", default="outputs",
                        help="Root directory containing responses_<model>/ subdirs")
    parser.add_argument("--output_dir", default="results/gold_error_analysis",
                        help="Output directory for results")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Specific tasks to analyze (default: all discovered)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to include (default: all 6 complete models)")
    args = parser.parse_args()

    models = args.models or MODELS
    responses_root = args.responses_root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover tasks from first model's response directory
    if args.tasks:
        tasks = args.tasks
    else:
        first_model_dir = Path(responses_root) / f"responses_{models[0]}"
        tasks = sorted(
            f.stem.replace("_responses", "")
            for f in first_model_dir.glob("*_responses.jsonl")
        )

    print(f"Models ({len(models)}): {', '.join(models)}")
    print(f"Tasks ({len(tasks)}): {', '.join(tasks)}")
    print(f"Thresholds: {', '.join(THRESHOLD_LABELS)}")
    print()

    # ── Per-task analysis ─────────────────────────────────────────────────────
    summary_rows = []
    all_flagged = defaultdict(list)  # threshold_label -> list of (task, record)

    for task in tasks:
        print(f"Analyzing {task}...", end=" ", flush=True)
        result = analyze_task(task, responses_root, models)

        if result is None:
            print("SKIP (no data)")
            continue

        total = result["total_samples"]
        n_models = result["n_models"]
        row = {"task": task, "total_samples": total, "n_models": n_models}

        for label in THRESHOLD_LABELS:
            flagged = result["flagged"][label]
            row[label] = len(flagged)
            for rec in flagged:
                all_flagged[label].append({"task": task, **rec})

        summary_rows.append(row)

        counts_str = " | ".join(f"{row[l]}" for l in THRESHOLD_LABELS)
        print(f"{total} samples | flagged: {counts_str}")

        # Save per-task flagged samples at highest resolution (33%)
        task_flagged = result["flagged"][THRESHOLD_LABELS[0]]
        if task_flagged:
            task_out = output_dir / f"{task}_flagged.jsonl"
            with open(task_out, "w") as f:
                for rec in task_flagged:
                    json.dump(rec, f)
                    f.write("\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"{'Task':<25} {'Samples':>8} {'Models':>7} | " +
          " | ".join(f"{l:>12}" for l in THRESHOLD_LABELS))
    print("-" * 90)

    for row in summary_rows:
        counts = " | ".join(f"{row.get(l, 0):>12}" for l in THRESHOLD_LABELS)
        print(f"{row['task']:<25} {row['total_samples']:>8} {row['n_models']:>7} | {counts}")

    print("=" * 90)

    # Totals
    total_samples = sum(r["total_samples"] for r in summary_rows)
    totals = {l: sum(r.get(l, 0) for r in summary_rows) for l in THRESHOLD_LABELS}
    counts_str = " | ".join(f"{totals[l]:>12}" for l in THRESHOLD_LABELS)
    print(f"{'TOTAL':<25} {total_samples:>8} {'':>7} | {counts_str}")
    print()

    # ── Save summary CSV ──────────────────────────────────────────────────────
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "total_samples", "n_models"] + THRESHOLD_LABELS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary saved to: {csv_path}")

    # ── Save all flagged samples per threshold ────────────────────────────────
    for label in THRESHOLD_LABELS:
        safe_label = label.replace("/", "_").replace(" ", "").replace("(", "").replace(")", "").replace("%", "pct")
        out_path = output_dir / f"flagged_threshold_{safe_label}.jsonl"
        with open(out_path, "w") as f:
            for rec in all_flagged[label]:
                json.dump(rec, f)
                f.write("\n")
        print(f"Flagged @{label}: {len(all_flagged[label])} samples → {out_path}")

    # ── Per-threshold pct summary ─────────────────────────────────────────────
    print()
    print("Flagged as % of total samples:")
    for label in THRESHOLD_LABELS:
        n = totals[label]
        pct = 100 * n / total_samples if total_samples > 0 else 0
        print(f"  {label}: {n:>5} / {total_samples} = {pct:.2f}%")


if __name__ == "__main__":
    main()
