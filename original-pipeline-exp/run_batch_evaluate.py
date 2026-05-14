#!/usr/bin/env python3
"""
Batch evaluation runner.

Iterates over all models in RAW_RESULTS_DIR, evaluates each JSONL file
in-process (no subprocess per task), and writes per-model JSON results
to OUTPUT_DIR/<model_name>/<task>_result.json.

Usage:
    python3 run_batch_evaluate.py [--raw_results_dir DIR] [--output_dir DIR]
                                  [--models MODEL [MODEL ...]] [--dry_run]
"""

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# External data for TAA evaluation
CTI_ALIAS_DICT = SCRIPT_DIR / "external/ctibench_eval/alias_dict.pickle"
CTI_RELATED_DICT = SCRIPT_DIR / "external/ctibench_eval/related_dict.pickle"
CTI_TAA_GT_TSV = SCRIPT_DIR / "external/ctibench_eval/cti_taa_gt.tsv"
ATHENA_ALIAS_CSV = SCRIPT_DIR / "external/athenabench_eval/aliases.csv"
ATHENA_RELATED_CSV = SCRIPT_DIR / "external/athenabench_eval/related_groups.csv"

# Map first-line `task` field value → benchmark name
TASK_ROUTING = {
    "ctibench_mcq":      "ctibench",
    "ctibench_rcm":      "ctibench",
    "ctibench_rcm_2021": "ctibench",
    "ctibench_vsp":      "ctibench",
    "ctibench_ate":      "ctibench",
    "ctibench_taa":      "ctibench",
    "athenabench_ckt":   "athenabench",
    "athenabench_ate":   "athenabench",
    "athenabench_rcm":   "athenabench",
    "athenabench_rms":   "athenabench",
    "athenabench_vsp":   "athenabench",
    "athenabench_taa":   "athenabench",
    "cybermetric":       "cybermetric",
    "seceval":           "seceval",
    "secure_maet":       "secure",
    "secure_cwet":       "secure",
    "secure_kcv":        "secure",
    "secbench_mcq":      "secbench",
    "secbench":          "secbench",
    "mmlu_cs":           "mmlu_cs",
}

TASK_PREFIX_ROUTING = {
    "redsage_":    "redsage",
    "cybermetric_": "cybermetric",
    "seceval_":    "seceval",
    "mmlu":        "mmlu_cs",
}

_ev = None  # cached evaluate module


def _load_evaluate():
    global _ev
    if _ev is None:
        spec = importlib.util.spec_from_file_location("evaluate", SCRIPT_DIR / "evaluate.py")
        _ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ev)
    return _ev


def detect_benchmark(jsonl_path: Path):
    try:
        with open(jsonl_path) as f:
            row = json.loads(f.readline())
    except Exception as e:
        return None, f"failed to read: {e}"

    task = row.get("task", "").lower().strip()
    if not task:
        return None, "no task field in first row"

    if task in TASK_ROUTING:
        return TASK_ROUTING[task], None

    for prefix, benchmark in TASK_PREFIX_ROUTING.items():
        if task.startswith(prefix):
            return benchmark, None

    return None, f"unrecognised task: {task!r}"


def run_evaluate_inprocess(jsonl_path: Path, benchmark: str,
                           output_path: Path, detail_path: Path,
                           dry_run: bool):
    if dry_run:
        print(f"  [DRY RUN] evaluate {jsonl_path.name} as {benchmark}")
        return True, None

    ev = _load_evaluate()

    kwargs = dict(
        input_jsonl=str(jsonl_path),
        detailed_output=str(detail_path),
    )

    try:
        if benchmark == "ctibench":
            raw = ev.evaluate_collected_ctibench(
                alias_dict_path=str(CTI_ALIAS_DICT),
                related_dict_path=str(CTI_RELATED_DICT),
                taa_gt_tsv_path=str(CTI_TAA_GT_TSV),
                **kwargs,
            )
        elif benchmark == "athenabench":
            raw = ev.evaluate_collected_athenabench(
                alias_csv_path=str(ATHENA_ALIAS_CSV),
                related_csv_path=str(ATHENA_RELATED_CSV),
                **kwargs,
            )
        elif benchmark == "cybermetric":
            raw = ev.evaluate_collected_cybermetric(**kwargs)
        elif benchmark == "seceval":
            raw = ev.evaluate_collected_seceval(**kwargs)
        elif benchmark == "redsage":
            raw = ev.evaluate_collected_redsage(**kwargs)
        elif benchmark == "secure":
            raw = ev.evaluate_collected_secure(**kwargs)
        elif benchmark == "cissp":
            raw = ev.evaluate_collected_cissp(**kwargs)
        elif benchmark == "secbench":
            raw = ev.evaluate_collected_secbench_mcq(**kwargs)
        elif benchmark == "mmlu_cs":
            raw = ev.evaluate_collected_mmlu_cs(**kwargs)
        else:
            return False, f"unsupported benchmark: {benchmark}"

        result = ev.build_unified_result(
            benchmark=benchmark,
            original_result=raw,
            input_jsonl=str(jsonl_path),
            detailed_output=str(detail_path),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return True, result

    except Exception as e:
        import traceback
        return False, traceback.format_exc()


def model_name_from_dir(d: Path) -> str:
    name = d.name
    return name[len("responses_"):] if name.startswith("responses_") else name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_results_dir",
        default=str(SCRIPT_DIR / "outputs"),
        help="Directory containing model subdirs (each with *_responses.jsonl files, "
             "either flat or in an inference_responses/ subdir)",
    )
    parser.add_argument(
        "--output_dir",
        default=str(SCRIPT_DIR / "outputs/eval_results"),
    )
    parser.add_argument("--models", nargs="*", default=None,
                        help="Limit to specific model names (default: all). "
                             "Match against the dir name with or without 'responses_' prefix.")
    parser.add_argument("--python", default=None,
                        help="Unused (kept for CLI compatibility). Evaluation runs in-process.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if output already exists")
    args = parser.parse_args()

    raw_dir = Path(args.raw_results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())

    if args.models:
        targets = {m.removeprefix("responses_") for m in args.models}
        all_dirs = [d for d in all_dirs if model_name_from_dir(d) in targets]

    summary = {}

    for model_dir in all_dirs:
        model_name = model_name_from_dir(model_dir)

        # Auto-detect layout: prefer inference_responses/ subdir, fall back to flat
        nested = model_dir / "inference_responses"
        if nested.exists() and any(nested.glob("*_responses.jsonl")):
            responses_dir = nested
        elif any(model_dir.glob("*_responses.jsonl")):
            responses_dir = model_dir
        else:
            print(f"[SKIP] {model_dir.name}: no *_responses.jsonl files found")
            continue

        model_out = out_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        summary[model_name] = {}

        print(f"\n=== {model_name} (from {model_dir.name}) ===", flush=True)

        jsonl_files = sorted(responses_dir.glob("*_responses.jsonl"))
        for jsonl_path in jsonl_files:
            stem = jsonl_path.stem.replace("_responses", "")

            benchmark, err = detect_benchmark(jsonl_path)
            if benchmark is None:
                print(f"  [SKIP] {jsonl_path.name}: {err}")
                continue

            output_path = model_out / f"{stem}_result.json"
            detail_path = model_out / f"{stem}_detail.jsonl"

            if output_path.exists() and not args.overwrite:
                print(f"  [EXISTS] {stem} ({benchmark})")
                try:
                    with open(output_path) as f:
                        result = json.load(f)
                    summary[model_name][stem] = result.get("primary_score")
                except Exception:
                    pass
                continue

            print(f"  Evaluating {stem} ({benchmark}) ...", end=" ", flush=True)
            ok, payload = run_evaluate_inprocess(
                jsonl_path, benchmark, output_path, detail_path, args.dry_run,
            )
            if ok:
                if args.dry_run:
                    print("OK  score=N/A")
                else:
                    score = payload.get("primary_score", "?")
                    summary[model_name][stem] = score
                    print(f"OK  score={score}", flush=True)
            else:
                print(f"FAIL\n    {payload}")
                summary[model_name][stem] = None

    # Write consolidated summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    # Print table
    all_tasks = sorted({t for m in summary.values() for t in m})
    all_models = sorted(summary)
    if all_tasks and all_models:
        col_w = max(len(t) for t in all_tasks) + 2
        mod_w = max(len(m) for m in all_models) + 2
        header = f"{'task':<{col_w}}" + "".join(f"{m[:mod_w-2]:<{mod_w}}" for m in all_models)
        print("\n" + header)
        print("-" * len(header))
        for task in all_tasks:
            row = f"{task:<{col_w}}"
            for model in all_models:
                val = summary.get(model, {}).get(task)
                if val is None:
                    cell = "N/A"
                elif isinstance(val, float):
                    cell = f"{val:.2f}"
                else:
                    cell = str(val)
                row += f"{cell:<{mod_w}}"
            print(row)


if __name__ == "__main__":
    main()
