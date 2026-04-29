#!/usr/bin/env python3
"""
Entry point for running RedSage MCQ tasks through the upstream RedSage
LightEval implementation with likelihood/logprob-based scoring.

Scoped to RedSage MCQ tasks only. Prompts, metrics, task configs, and all
scoring logic live entirely in external/RedSage/eval/cybersecurity_benchmarks.py
and are not reimplemented here.

LightEval result files are written as-is by LightEval under the output directory.
This script writes a manifest.json recording the run configuration in the same
metadata style used by run_inference_benchmarks.py.

Task modes:
  logprob     5 redsage_mcq:* subsets, loglikelihood scoring  [default]
  generative  5 redsage_mcq_em:* subsets, exact-match scoring

Usage:
    # All logprob MCQ subsets (default task mode)
    python run_redsage_lighteval.py --model RISys-Lab/RedSage-Qwen3-8B-Ins

    # Generative (exact-match) variants
    python run_redsage_lighteval.py --model ... --task-mode generative

    # vLLM backend, smoke test
    python run_redsage_lighteval.py vllm --model ... --max-samples 5

    # Specific subsets only (logprob by default)
    python run_redsage_lighteval.py --model ... --subsets cybersecurity_skills,cybersecurity_tools_kali

Setup:
    cd external/RedSage/eval/lighteval && pip install -e . && cd ../../../../
    pip install cvss aenum
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.absolute()
REDSAGE_EVAL_DIR = REPO_ROOT / "external" / "RedSage" / "eval"
CUSTOM_TASKS_PATH = REDSAGE_EVAL_DIR / "cybersecurity_benchmarks.py"

KNOWN_SUBSETS = [
    "cybersecurity_knowledge_generals",
    "cybersecurity_knowledge_frameworks",
    "cybersecurity_skills",
    "cybersecurity_tools_cli",
    "cybersecurity_tools_kali",
]

_TASK_PREFIX = {
    "logprob":    "redsage_mcq",
    "generative": "redsage_mcq_em",
}


def resolve_tasks(task_mode: str, subsets: list[str], num_fewshot: int) -> list[str]:
    prefix = _TASK_PREFIX[task_mode]
    return [f"{prefix}:{s}|{num_fewshot}" for s in subsets]


def build_model_args(args: argparse.Namespace) -> str:
    if args.backend == "vllm":
        parts = [
            f"model_name={args.model}",
            f"gpu_memory_utilization={args.vllm_gpu_memory_utilization}",
            f"tensor_parallel_size={args.vllm_tensor_parallel_size}",
        ]
        if args.max_model_len:
            parts.append(f"max_model_length={args.max_model_len}")
        if args.use_chat_template:
            parts.append("override_chat_template=True")
        elif args.no_chat_template:
            parts.append("override_chat_template=False")
    else:
        parts = [f"model_name={args.model}"]
        if args.max_model_len:
            parts.append(f"max_length={args.max_model_len}")
        if args.use_chat_template:
            parts.append("override_chat_template=True")
        elif args.no_chat_template:
            parts.append("override_chat_template=False")
    return ",".join(parts)


def build_lighteval_cmd(args: argparse.Namespace, tasks_str: str, output_dir: str) -> list[str]:
    cmd = ["lighteval", args.backend, build_model_args(args)]
    if args.save_details:
        cmd.append("--save-details")
    cmd.extend([
        "--custom-tasks", str(CUSTOM_TASKS_PATH),
        "--output-dir", output_dir,
        tasks_str,
    ])
    if args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run RedSage MCQ tasks through the upstream RedSage LightEval implementation. "
            "Scoped to redsage_mcq (logprob) and redsage_mcq_em (generative) tasks only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Task modes (--task-mode):
  logprob     redsage_mcq:* subsets with loglikelihood scoring  [default]
  generative  redsage_mcq_em:* subsets with exact-match scoring

Known subsets (--subsets):
  {chr(10).join("  " + s for s in KNOWN_SUBSETS)}
  Default: all five subsets.

Output:
  LightEval writes its result files under --output-dir.
  A manifest.json is written at the root of --output-dir with run metadata.

Setup:
  cd external/RedSage/eval/lighteval && pip install -e . && cd ../../../../
  pip install cvss aenum
        """,
    )

    parser.add_argument(
        "backend",
        nargs="?",
        default="accelerate",
        choices=["accelerate", "vllm"],
        help="Inference backend (default: accelerate)",
    )
    parser.add_argument(
        "--model", required=True,
        help="HF model name or local path (e.g. RISys-Lab/RedSage-Qwen3-8B-Ins)",
    )
    parser.add_argument(
        "--task-mode",
        default="logprob",
        choices=["logprob", "generative"],
        help="Scoring mode: logprob (loglikelihood) or generative (exact-match) (default: logprob)",
    )
    parser.add_argument(
        "--subsets",
        default=None,
        help=(
            "Comma-separated subset names to run (default: all five). "
            f"Choices: {', '.join(KNOWN_SUBSETS)}"
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Auto-generated as results/redsage_<model>_<timestamp> if omitted.",
    )
    parser.add_argument(
        "--num-fewshot", type=int, default=0,
        help="Few-shot examples per task (default: 0)",
    )
    parser.add_argument(
        "--max-samples", type=int,
        help="Max samples per task (useful for smoke tests)",
    )
    parser.add_argument(
        "--save-details", action="store_true",
        help="Save per-sample predictions (passed through to LightEval)",
    )

    chat_group = parser.add_mutually_exclusive_group()
    chat_group.add_argument(
        "--use-chat-template", action="store_true",
        help="Force chat template on (recommended for instruction-tuned models)",
    )
    chat_group.add_argument(
        "--no-chat-template", action="store_true",
        help="Force chat template off (recommended for base models)",
    )

    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--max-model-len", "--vllm-max-model-len",
        dest="max_model_len", type=int,
        help="Maximum model sequence length",
    )

    args = parser.parse_args()

    # Validate external dependency
    if not CUSTOM_TASKS_PATH.exists():
        print(
            f"Error: RedSage custom tasks not found at {CUSTOM_TASKS_PATH}\n"
            "Ensure external/RedSage/ is present (git submodule or manual copy).",
            file=sys.stderr,
        )
        return 1

    # Resolve and validate subsets
    if args.subsets:
        requested = [s.strip() for s in args.subsets.split(",") if s.strip()]
        unknown = [s for s in requested if s not in KNOWN_SUBSETS]
        if unknown:
            print(
                f"Error: unknown subset(s): {unknown}\n"
                f"Known subsets: {KNOWN_SUBSETS}",
                file=sys.stderr,
            )
            return 1
        subsets = requested
    else:
        subsets = list(KNOWN_SUBSETS)

    # Resolve lighteval task strings
    task_list = resolve_tasks(args.task_mode, subsets, args.num_fewshot)
    tasks_str = ",".join(task_list)

    # Determine output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_slug = args.model.rstrip("/").split("/")[-1].replace("\\", "-")
        output_dir = f"results/redsage_{model_slug}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Build lighteval command
    cmd = build_lighteval_cmd(args, tasks_str, output_dir)
    cmd_str = " ".join(cmd)

    # Write manifest before running (documents the run configuration)
    manifest = {
        "model": args.model,
        "backend": args.backend,
        "task_mode": args.task_mode,
        "subsets": subsets,
        "resolved_tasks": task_list,
        "output_dir": output_dir,
        "lighteval_command": cmd_str,
        "timestamp": datetime.now().isoformat(),
        "num_fewshot": args.num_fewshot,
        "max_samples": args.max_samples,
        "save_details": args.save_details,
        "upstream_redsage_eval_dir": str(REDSAGE_EVAL_DIR),
        "upstream_custom_tasks": str(CUSTOM_TASKS_PATH),
        "scoring_note": (
            "Loglikelihood scoring" if args.task_mode == "logprob"
            else "Exact-match / generative scoring"
        ) + (
            " performed by LightEval using the upstream RedSage RedSageMCQTask config "
            "from external/RedSage/eval/cybersecurity_benchmarks.py. "
            "LightEval result files are written as-is under output_dir."
        ),
        "status": "started",
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("RedSage MCQ LightEval run")
    print(f"  model:      {args.model}")
    print(f"  backend:    {args.backend}")
    print(f"  task mode:  {args.task_mode}")
    print(f"  subsets:    {subsets}")
    print(f"  output dir: {output_dir}")
    print(f"  manifest:   {manifest_path}")
    print()
    print("lighteval command:")
    print(" ", cmd_str)
    print("=" * 80 + "\n")

    try:
        result = subprocess.run(cmd, check=True)
        exit_code = result.returncode
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        print(f"\nError: lighteval exited with code {exit_code}", file=sys.stderr)
    except FileNotFoundError:
        print(
            "\nError: lighteval not found. Install from the vendored submodule:\n"
            "  cd external/RedSage/eval/lighteval && pip install -e . && cd ../../../../\n"
            "  pip install cvss aenum",
            file=sys.stderr,
        )
        return 1

    # Update manifest with final status
    manifest["status"] = "completed" if exit_code == 0 else f"failed (exit code {exit_code})"
    manifest["completed_at"] = datetime.now().isoformat()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
