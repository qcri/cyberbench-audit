"""Step 2 — Gold-error voting studies.

Reads `outputs/responses_<model>/<task>_responses.jsonl` for every model and
sub-task, extracts a normalized prediction per sample, and runs four majority-
voting analyses to flag samples whose gold answer is suspect.

Scope: only votable sub-tasks (MCQ, ID, VSP). Free-form text tasks (rcm, taa,
athena_rcm) are skipped.

Outputs (under analysis/reports/):
  gold_errors/
    threshold_sweep.csv     rows = sub-tasks, cols = thresholds, value = #flagged
    threshold_sweep.md
    flagged/<task>_t050.jsonl
  weighted_rank/
    summary.csv             rows = sub-tasks, cols = {linear, harmonic} #flagged @ 0.50
    flagged/<task>_w<scheme>.jsonl
  topk/
    summary.csv             rows = sub-tasks, cols = k=1..n_models
    flagged/<task>_k<k>.jsonl
  acceptance/
    summary.csv             rows = sub-tasks, cols = alpha (cutoff = best * alpha)
    flagged/<task>_a<alpha>.jsonl

Run from BenchmarkingSecBenchmarks/:
  python -m analysis.gold_error_voting
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from analysis.lib.extraction import (
    VOTABLE_TASKS,
    extract_prediction,
    normalize_gold,
    normalize_task,
)
from analysis.lib.loaders import (
    TASK_ORDER,
    discover_models,
    iter_jsonl,
    responses_path,
)
from analysis.lib.voting import (
    VoteResult,
    acceptance_filter,
    harmonic_rank_weights,
    is_flagged,
    linear_rank_weights,
    topk_filter,
    vote,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
REPORTS_DIR = HERE / "reports"

ACCURACY_CACHE = REPORTS_DIR / "per_model_task_accuracy.json"

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 2 / 3, 0.75, 5 / 6, 0.90, 1.00]
THRESHOLD_LABELS = ["0.30", "0.40", "0.50", "0.60", "0.667", "0.75", "0.833", "0.90", "1.00"]
DEFAULT_DUMP_THRESHOLD = 0.50

ACCEPTANCE_ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9]
ACCEPTANCE_MIN_QUORUM = 3

DEFAULT_VOTE_THRESHOLD = 0.50  # for weighted/topk/acceptance flagging


def safe_label(s: str) -> str:
    return s.replace("/", "_").replace(".", "p")


def record_to_dict(task: str, idx: str, vr: VoteResult, question: str) -> dict:
    return {
        "task": task,
        "index": idx,
        "gold": vr.gold,
        "majority_prediction": vr.majority_prediction,
        "agreement_count": round(vr.agreement_count, 4),
        "agreement_total": round(vr.agreement_total, 4),
        "agreement_fraction": round(vr.agreement_fraction, 4),
        "n_models_with_data": vr.n_models_with_data,
        "agreeing_models": vr.agreeing_models,
        "all_predictions": vr.all_predictions,
        "question_snippet": (question or "")[:300],
    }


def build_task_matrix(
    task: str, models: List[str]
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, str]]:
    """Return (predictions_by_idx, gold_by_idx, question_by_idx).

    predictions_by_idx[idx][model] = normalized prediction (only models with
    non-null prediction for that sample appear).
    """
    predictions: Dict[str, Dict[str, str]] = {}
    gold_by_idx: Dict[str, str] = {}
    question_by_idx: Dict[str, str] = {}

    for model in models:
        path = responses_path(OUTPUTS_ROOT, model, task)
        if not path.exists():
            continue
        for sample in iter_jsonl(path):
            idx = str(sample.get("index", ""))
            if not idx:
                continue
            if idx not in gold_by_idx:
                gold_by_idx[idx] = normalize_gold(sample.get("ground_truth", ""), task)
                q = sample.get("prompt", "")
                if isinstance(q, list):
                    q = next(
                        (msg.get("content", "")
                         for msg in reversed(q)
                         if isinstance(msg, dict) and msg.get("role") == "user"),
                        str(q),
                    )
                question_by_idx[idx] = q[:300] if isinstance(q, str) else str(q)[:300]
            pred = extract_prediction(sample.get("model_response", ""), task)
            if pred is None:
                continue
            predictions.setdefault(idx, {})[model] = pred

    return predictions, gold_by_idx, question_by_idx


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            json.dump(r, f)
            f.write("\n")


def votable(task: str) -> bool:
    return normalize_task(task) in VOTABLE_TASKS


def run_threshold_sweep(
    task: str,
    matrix: Dict[str, Dict[str, str]],
    gold_by_idx: Dict[str, str],
    question_by_idx: Dict[str, str],
    out_dir: Path,
) -> Dict[str, int]:
    """Plain unweighted majority across `THRESHOLDS`."""
    flagged_counts = {label: 0 for label in THRESHOLD_LABELS}
    dump_records: List[dict] = []
    for idx, preds in matrix.items():
        gold = gold_by_idx.get(idx)
        if gold is None:
            continue
        vr = vote(preds, gold)
        if vr is None:
            continue
        for thr, label in zip(THRESHOLDS, THRESHOLD_LABELS):
            if is_flagged(vr, thr):
                flagged_counts[label] += 1
        if is_flagged(vr, DEFAULT_DUMP_THRESHOLD):
            dump_records.append(record_to_dict(task, idx, vr, question_by_idx.get(idx, "")))

    if dump_records:
        write_jsonl(out_dir / "flagged" / f"{task}_t{int(DEFAULT_DUMP_THRESHOLD * 100):02d}.jsonl", dump_records)
    return flagged_counts


def run_weighted(
    task: str,
    matrix: Dict[str, Dict[str, str]],
    gold_by_idx: Dict[str, str],
    question_by_idx: Dict[str, str],
    accuracies: Dict[str, float],
    out_dir: Path,
) -> Dict[str, int]:
    """Rank-weighted majority with linear and harmonic schemes."""
    schemes = {
        "linear": linear_rank_weights(accuracies),
        "harmonic": harmonic_rank_weights(accuracies),
    }
    flagged_counts = {scheme: 0 for scheme in schemes}
    dumps: Dict[str, List[dict]] = {scheme: [] for scheme in schemes}

    for idx, preds in matrix.items():
        gold = gold_by_idx.get(idx)
        if gold is None:
            continue
        for scheme, weights in schemes.items():
            scheme_preds = {m: p for m, p in preds.items() if m in weights}
            vr = vote(scheme_preds, gold, weights=weights)
            if vr is None:
                continue
            if is_flagged(vr, DEFAULT_VOTE_THRESHOLD):
                flagged_counts[scheme] += 1
                dumps[scheme].append(record_to_dict(task, idx, vr, question_by_idx.get(idx, "")))

    for scheme, recs in dumps.items():
        if recs:
            write_jsonl(out_dir / "flagged" / f"{task}_w{scheme}.jsonl", recs)
    return flagged_counts


def run_topk(
    task: str,
    matrix: Dict[str, Dict[str, str]],
    gold_by_idx: Dict[str, str],
    question_by_idx: Dict[str, str],
    accuracies: Dict[str, float],
    out_dir: Path,
    ks: List[int],
) -> Dict[int, int]:
    flagged_counts = {k: 0 for k in ks}
    dumps: Dict[int, List[dict]] = {k: [] for k in ks}

    for idx, preds in matrix.items():
        gold = gold_by_idx.get(idx)
        if gold is None:
            continue
        for k in ks:
            sub = topk_filter(preds, accuracies, k)
            vr = vote(sub, gold)
            if vr is None:
                continue
            if is_flagged(vr, DEFAULT_VOTE_THRESHOLD):
                flagged_counts[k] += 1
                dumps[k].append(record_to_dict(task, idx, vr, question_by_idx.get(idx, "")))

    for k, recs in dumps.items():
        if recs:
            write_jsonl(out_dir / "flagged" / f"{task}_k{k}.jsonl", recs)
    return flagged_counts


def run_acceptance(
    task: str,
    matrix: Dict[str, Dict[str, str]],
    gold_by_idx: Dict[str, str],
    question_by_idx: Dict[str, str],
    accuracies: Dict[str, float],
    out_dir: Path,
) -> Tuple[Dict[float, int], Dict[float, int]]:
    """Return (flagged_per_alpha, n_eligible_per_alpha)."""
    if not accuracies:
        return ({a: 0 for a in ACCEPTANCE_ALPHAS}, {a: 0 for a in ACCEPTANCE_ALPHAS})

    best = max(accuracies.values())
    flagged_counts = {a: 0 for a in ACCEPTANCE_ALPHAS}
    eligible_counts = {a: 0 for a in ACCEPTANCE_ALPHAS}
    dumps: Dict[float, List[dict]] = {a: [] for a in ACCEPTANCE_ALPHAS}

    for alpha in ACCEPTANCE_ALPHAS:
        cutoff = best * alpha
        eligible = {m for m, a in accuracies.items() if a >= cutoff}
        eligible_counts[alpha] = len(eligible)
        if len(eligible) < ACCEPTANCE_MIN_QUORUM:
            flagged_counts[alpha] = -1  # sentinel: skipped
            continue
        for idx, preds in matrix.items():
            gold = gold_by_idx.get(idx)
            if gold is None:
                continue
            sub = acceptance_filter(preds, accuracies, cutoff)
            vr = vote(sub, gold)
            if vr is None:
                continue
            if is_flagged(vr, DEFAULT_VOTE_THRESHOLD):
                flagged_counts[alpha] += 1
                dumps[alpha].append(record_to_dict(task, idx, vr, question_by_idx.get(idx, "")))

    for alpha, recs in dumps.items():
        if recs:
            write_jsonl(out_dir / "flagged" / f"{task}_a{safe_label(str(alpha))}.jsonl", recs)
    return flagged_counts, eligible_counts


PARTIAL_DIR_NAME = ".partial"


def process_task(task: str, models: List[str], cache: dict, partial_dir: Path,
                 sweep_dir: Path, weighted_dir: Path, topk_dir: Path,
                 acceptance_dir: Path, ks: List[int]) -> None:
    """Compute per-task voting summaries and persist a partial JSON."""
    matrix, gold_by_idx, question_by_idx = build_or_load_matrix(task, models, partial_dir)
    n_samples = len(gold_by_idx)
    if not matrix:
        return

    accuracies = {
        m: cache.get(m, {}).get(task)
        for m in models
        if cache.get(m, {}).get(task) is not None
    }

    sweep = run_threshold_sweep(task, matrix, gold_by_idx, question_by_idx, sweep_dir)
    weighted = run_weighted(task, matrix, gold_by_idx, question_by_idx, accuracies, weighted_dir)
    topk = run_topk(task, matrix, gold_by_idx, question_by_idx, accuracies, topk_dir, ks)
    flag_a, elig_a = run_acceptance(task, matrix, gold_by_idx, question_by_idx, accuracies, acceptance_dir)

    payload = {
        "task": task,
        "n_samples": n_samples,
        "best_acc": round(max(accuracies.values()), 4) if accuracies else 0,
        "sweep": sweep,
        "weighted": weighted,
        "topk": topk,
        "acceptance_flag": {str(a): v for a, v in flag_a.items()},
        "acceptance_eligible": {str(a): v for a, v in elig_a.items()},
    }
    partial_dir.mkdir(parents=True, exist_ok=True)
    (partial_dir / f"{task}.json").write_text(json.dumps(payload))


def matrix_cache_path(task: str, partial_dir: Path) -> Path:
    return partial_dir / "chunks" / task / "_matrix.json"


def build_or_load_matrix(task: str, models: List[str], partial_dir: Path):
    """Read matrix from cache if it exists, else build and cache it."""
    cache_path = matrix_cache_path(task, partial_dir)
    if cache_path.exists():
        d = json.loads(cache_path.read_text())
        return d["matrix"], d["gold"], d["question"]
    matrix, gold, question = build_task_matrix(task, models)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"matrix": matrix, "gold": gold, "question": question}))
    return matrix, gold, question


def process_task_chunk(task: str, models: List[str], cache: dict, partial_dir: Path,
                        sweep_dir: Path, weighted_dir: Path, topk_dir: Path,
                        acceptance_dir: Path, ks: List[int],
                        chunk_idx: int, n_chunks: int) -> None:
    """Process just one slice of the samples for a task. Saves to .chunk_<i>.json.

    Use the `aggregate_chunks` step afterwards to merge chunks for the task into
    a normal partial JSON.
    """
    matrix, gold_by_idx, question_by_idx = build_or_load_matrix(task, models, partial_dir)
    if not matrix:
        return

    sample_ids = sorted(matrix.keys(), key=lambda x: int(x) if x.isdigit() else x)
    total = len(sample_ids)
    start = (total * chunk_idx) // n_chunks
    end = (total * (chunk_idx + 1)) // n_chunks
    keep = set(sample_ids[start:end])
    matrix = {idx: preds for idx, preds in matrix.items() if idx in keep}
    gold_by_idx = {idx: g for idx, g in gold_by_idx.items() if idx in keep}
    question_by_idx = {idx: q for idx, q in question_by_idx.items() if idx in keep}

    accuracies = {
        m: cache.get(m, {}).get(task)
        for m in models
        if cache.get(m, {}).get(task) is not None
    }

    chunk_dir = partial_dir / "chunks" / task
    chunk_dir.mkdir(parents=True, exist_ok=True)
    sweep = run_threshold_sweep(task, matrix, gold_by_idx, question_by_idx,
                                 chunk_dir / f"sweep_{chunk_idx}")
    weighted = run_weighted(task, matrix, gold_by_idx, question_by_idx, accuracies,
                             chunk_dir / f"weighted_{chunk_idx}")
    topk = run_topk(task, matrix, gold_by_idx, question_by_idx, accuracies,
                     chunk_dir / f"topk_{chunk_idx}", ks)
    flag_a, elig_a = run_acceptance(task, matrix, gold_by_idx, question_by_idx, accuracies,
                                     chunk_dir / f"acceptance_{chunk_idx}")

    payload = {
        "task": task,
        "chunk_idx": chunk_idx,
        "n_chunks": n_chunks,
        "n_samples": len(matrix),
        "best_acc": round(max(accuracies.values()), 4) if accuracies else 0,
        "sweep": sweep,
        "weighted": weighted,
        "topk": topk,
        "acceptance_flag": {str(a): v for a, v in flag_a.items()},
        "acceptance_eligible": {str(a): v for a, v in elig_a.items()},
    }
    (chunk_dir / f"chunk_{chunk_idx}.json").write_text(json.dumps(payload))


def merge_chunks_for_task(task: str, partial_dir: Path) -> None:
    """Merge chunk_*.json files for a task into a single partial JSON."""
    chunk_dir = partial_dir / "chunks" / task
    chunks = sorted(chunk_dir.glob("chunk_*.json"))
    if not chunks:
        return
    payloads = [json.loads(p.read_text()) for p in chunks]
    n_chunks = max(p["n_chunks"] for p in payloads)
    merged = {
        "task": task,
        "n_samples": sum(p["n_samples"] for p in payloads),
        "best_acc": payloads[0]["best_acc"],
    }
    for key in ["sweep", "weighted", "topk", "acceptance_flag"]:
        agg = {}
        for p in payloads:
            for k, v in p[key].items():
                if v == -1:  # skipped sentinel — preserve
                    agg[k] = -1
                elif agg.get(k) == -1:
                    continue
                else:
                    agg[k] = agg.get(k, 0) + v
        merged[key] = agg
    # eligibility is a per-task constant — take it from the first chunk.
    merged["acceptance_eligible"] = payloads[0]["acceptance_eligible"]
    (partial_dir / f"{task}.json").write_text(json.dumps(merged))


def aggregate(models: List[str], partial_dir: Path,
              sweep_dir: Path, weighted_dir: Path, topk_dir: Path,
              acceptance_dir: Path, ks: List[int]) -> None:
    """Read all partial JSONs and write the final CSVs / MD."""
    sweep_rows = []
    weighted_rows = []
    topk_rows = []
    acceptance_flag_rows = []
    acceptance_eligible_rows = []

    for task in TASK_ORDER:
        if not votable(task):
            continue
        path = partial_dir / f"{task}.json"
        if not path.exists():
            continue
        p = json.loads(path.read_text())
        sweep_rows.append({"task": p["task"], "n_samples": p["n_samples"], **p["sweep"]})
        weighted_rows.append({"task": p["task"], "n_samples": p["n_samples"], **p["weighted"]})
        topk_rows.append({"task": p["task"], "n_samples": p["n_samples"],
                          **{f"k={k}": p["topk"].get(str(k), p["topk"].get(k, 0)) for k in ks}})
        acceptance_flag_rows.append({
            "task": p["task"], "n_samples": p["n_samples"], "best_acc": p["best_acc"],
            **{f"alpha={a}": p["acceptance_flag"].get(str(a), 0) for a in ACCEPTANCE_ALPHAS},
        })
        acceptance_eligible_rows.append({
            "task": p["task"], "n_samples": p["n_samples"],
            **{f"alpha={a}": p["acceptance_eligible"].get(str(a), 0) for a in ACCEPTANCE_ALPHAS},
        })

    _write_outputs(models, sweep_rows, weighted_rows, topk_rows,
                   acceptance_flag_rows, acceptance_eligible_rows,
                   sweep_dir, weighted_dir, topk_dir, acceptance_dir, ks)


def _write_outputs(models, sweep_rows, weighted_rows, topk_rows,
                   acceptance_flag_rows, acceptance_eligible_rows,
                   sweep_dir, weighted_dir, topk_dir, acceptance_dir, ks):
    # ---------------- write CSVs ----------------
    with open(sweep_dir / "threshold_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "n_samples", *THRESHOLD_LABELS])
        for r in sweep_rows:
            w.writerow([r["task"], r["n_samples"], *(r[label] for label in THRESHOLD_LABELS)])

    with open(weighted_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "n_samples", "linear", "harmonic"])
        for r in weighted_rows:
            w.writerow([r["task"], r["n_samples"], r["linear"], r["harmonic"]])

    with open(topk_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "n_samples", *(f"k={k}" for k in ks)])
        for r in topk_rows:
            w.writerow([r["task"], r["n_samples"], *(r[f"k={k}"] for k in ks)])

    with open(acceptance_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "n_samples", "best_acc", *(f"alpha={a}" for a in ACCEPTANCE_ALPHAS)])
        for r in acceptance_flag_rows:
            w.writerow([
                r["task"], r["n_samples"], r["best_acc"],
                *(r[f"alpha={a}"] for a in ACCEPTANCE_ALPHAS),
            ])

    with open(acceptance_dir / "eligible_models.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "n_samples", *(f"alpha={a}" for a in ACCEPTANCE_ALPHAS)])
        for r in acceptance_eligible_rows:
            w.writerow([r["task"], r["n_samples"], *(r[f"alpha={a}"] for a in ACCEPTANCE_ALPHAS)])

    # ---------------- markdown summary for sweep ----------------
    md = ["# Threshold sweep — # samples flagged per sub-task", "",
          "Rows = votable sub-tasks. Columns = agreement-fraction thresholds. "
          "A sample is flagged when ≥X% of voting models agree on a prediction "
          "different from the gold answer.",
          "",
          "| Task | N | " + " | ".join(THRESHOLD_LABELS) + " |",
          "|" + "|".join(["---"] * (2 + len(THRESHOLD_LABELS))) + "|"]
    for r in sweep_rows:
        md.append(f"| {r['task']} | {r['n_samples']} | " +
                  " | ".join(str(r[label]) for label in THRESHOLD_LABELS) + " |")
    md.append("")
    (sweep_dir / "threshold_sweep.md").write_text("\n".join(md))

    print(
        f"gold_error_voting: {len(sweep_rows)} votable tasks aggregated. "
        f"Outputs in {REPORTS_DIR}/{{gold_errors,weighted_rank,topk,acceptance}}/."
    )


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Process only this task (chunked mode); writes partial JSON.")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate partial JSONs into final CSVs.")
    parser.add_argument("--chunk-idx", type=int, default=None,
                        help="With --task: process only chunk i of n.")
    parser.add_argument("--n-chunks", type=int, default=None,
                        help="With --task: total chunks for this task.")
    parser.add_argument("--merge-chunks", help="Merge chunk_*.json for the given task.")
    parser.add_argument("--prepare-matrix", help="Build & cache the prediction matrix for a task without voting.")
    args = parser.parse_args(argv)

    if not ACCURACY_CACHE.exists():
        raise SystemExit(
            f"Missing {ACCURACY_CACHE}. Run `python -m analysis.results_table` first."
        )
    cache = json.loads(ACCURACY_CACHE.read_text())
    models = discover_models(OUTPUTS_ROOT)

    sweep_dir = REPORTS_DIR / "gold_errors"
    weighted_dir = REPORTS_DIR / "weighted_rank"
    topk_dir = REPORTS_DIR / "topk"
    acceptance_dir = REPORTS_DIR / "acceptance"
    partial_dir = REPORTS_DIR / PARTIAL_DIR_NAME
    for d in [sweep_dir, weighted_dir, topk_dir, acceptance_dir]:
        (d / "flagged").mkdir(parents=True, exist_ok=True)

    n_models = len(models)
    ks = list(range(1, n_models + 1))

    if args.aggregate:
        aggregate(models, partial_dir, sweep_dir, weighted_dir, topk_dir, acceptance_dir, ks)
        return

    if args.merge_chunks:
        merge_chunks_for_task(args.merge_chunks, partial_dir)
        print(f"Merged chunks → {partial_dir / (args.merge_chunks + '.json')}")
        return

    if args.prepare_matrix:
        task_name = args.prepare_matrix
        cache_path = matrix_cache_path(task_name, partial_dir)
        if cache_path.exists():
            print(f"Matrix already cached at {cache_path}")
            return
        # Incremental build: process one model at a time, persisting progress.
        progress_path = cache_path.parent / "_matrix_progress.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if progress_path.exists():
            state = json.loads(progress_path.read_text())
        else:
            state = {"done_models": [], "predictions": {}, "gold": {}, "question": {}}
        for model in models:
            if model in state["done_models"]:
                continue
            path = responses_path(OUTPUTS_ROOT, model, task_name)
            if path.exists():
                for sample in iter_jsonl(path):
                    idx = str(sample.get("index", ""))
                    if not idx:
                        continue
                    if idx not in state["gold"]:
                        state["gold"][idx] = normalize_gold(sample.get("ground_truth", ""), task_name)
                        q = sample.get("prompt", "")
                        if isinstance(q, list):
                            q = next(
                                (msg.get("content", "")
                                 for msg in reversed(q)
                                 if isinstance(msg, dict) and msg.get("role") == "user"),
                                str(q),
                            )
                        state["question"][idx] = (q[:300] if isinstance(q, str) else str(q)[:300])
                    pred = extract_prediction(sample.get("model_response", ""), task_name)
                    if pred is not None:
                        state["predictions"].setdefault(idx, {})[model] = pred
            state["done_models"].append(model)
            progress_path.write_text(json.dumps(state))
            print(f"  loaded {model}: {len(state['predictions'])} samples so far", flush=True)
        cache_path.write_text(json.dumps({
            "matrix": state["predictions"],
            "gold": state["gold"],
            "question": state["question"],
        }))
        progress_path.unlink(missing_ok=True)
        print(f"Cached matrix for '{task_name}': {len(state['predictions'])} samples → {cache_path}")
        return

    if args.task:
        if not votable(args.task):
            print(f"Task '{args.task}' is not votable; skipping.")
            return
        if args.chunk_idx is not None and args.n_chunks is not None:
            process_task_chunk(args.task, models, cache, partial_dir,
                               sweep_dir, weighted_dir, topk_dir, acceptance_dir, ks,
                               args.chunk_idx, args.n_chunks)
            print(f"Wrote chunk {args.chunk_idx}/{args.n_chunks} of '{args.task}'")
            return
        process_task(args.task, models, cache, partial_dir,
                     sweep_dir, weighted_dir, topk_dir, acceptance_dir, ks)
        print(f"Wrote {partial_dir / (args.task + '.json')}")
        return

    # Full run: process every task then aggregate.
    for task in TASK_ORDER:
        if not votable(task):
            continue
        process_task(task, models, cache, partial_dir,
                     sweep_dir, weighted_dir, topk_dir, acceptance_dir, ks)
    aggregate(models, partial_dir, sweep_dir, weighted_dir, topk_dir, acceptance_dir, ks)


if __name__ == "__main__":
    main()
