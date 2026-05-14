#!/usr/bin/env python3
"""
Token Calibration Script — BenchBench

Determines optimal per-task max_tokens values by running a small sample of
each benchmark task and measuring actual output lengths. This prevents
responses from being cut off mid-answer while avoiding wastefully large
token budgets.

Output JSON format matches slurm/calibration_*.json.

Usage:
    python calibrate_max_tokens.py \\
        --model_path /path/to/model \\
        --n_samples 10 \\
        --ceiling 2048 \\
        --gpu_mem 0.90 \\
        --max_model_len 8192 \\
        --output slurm/calibration_MyModel.json \\
        [--tasks mcq rcm vsp ...]  \\
        [--is_thinking_model]
"""

import os
import re
import json
import math
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# Import dataset loaders from the main inference script
from run_inference_benchmarks import (
    get_task_type,
    load_jsonl_dataset,
    collect_mmlu_cs,          # imported for reference; we re-use load logic below
)
from datasets import load_dataset


# ── Default task list ──────────────────────────────────────────────────────────
ALL_TASKS = [
    "mcq", "rcm", "vsp", "ate",                        # CTI-Bench (cti_taa not in HF mirror)
    "ckt", "rms", "taa",                              # AthenaBench classic
    "athena_ate", "athena_rcm", "athena_vsp",          # AthenaBench expanded
    "secure_maet", "secure_cwet", "secure_kcv",        # SECURE
    "seceval",                                         # SecEval
    "cybermetric",                                     # CyberMetric-500
    "mmlu-cs",                                         # MMLU Computer Security
    "secbench",                                        # SecBench
    "redsage_frameworks", "redsage_generals",          # RedSage-MCQ
    "redsage_skills", "redsage_cli", "redsage_kali",
]

# Dataset source map (mirrors task_map in run_inference_benchmarks.py)
TASK_MAP = {
    "mcq":      ("hf", "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-mcq"),
    "rcm":      ("hf", "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-rcm"),
    "vsp":      ("hf", "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-vsp"),
    "ate":      ("hf", "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-ate"),
    # cti_taa: not in HF mirror; use AthenaBench taa instead
    "mmlu-cs":  ("hf", "lighteval/mmlu", "computer_security"),
    "secure_maet": ("hf", "RISys-Lab/Benchmarks_CyberSec_SECURE", "MAET"),
    "secure_cwet": ("hf", "RISys-Lab/Benchmarks_CyberSec_SECURE", "CWET"),
    "secure_kcv":  ("hf", "RISys-Lab/Benchmarks_CyberSec_SECURE", "KCV"),
    "secbench": ("hf", "RISys-Lab/Benchmarks_CyberSec_SecBench", "MCQs_English"),
    "redsage_frameworks": ("hf", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_knowledge_frameworks"),
    "redsage_generals":   ("hf", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_knowledge_generals"),
    "redsage_skills":     ("hf", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_skills"),
    "redsage_cli":        ("hf", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_tools_cli"),
    "redsage_kali":       ("hf", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_tools_kali"),
    "ckt":      ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-ckt-3k.jsonl", None),
    "rms":      ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-rms.jsonl", None),
    "taa":      ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-taa.jsonl", None),
    "athena_ate": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-ate.jsonl", None),
    "athena_rcm": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-rcm.jsonl", None),
    "athena_vsp": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-vsp.jsonl", None),
    # Special loaders (handled in load_sample_prompts)
    "seceval":    ("special", "https://huggingface.co/datasets/XuanwuAI/SecEval/resolve/main/questions.json", None),
    "cybermetric": ("hf", "RISys-Lab/Benchmarks_CyberSec_CyberMetrics", "cyberMetric_500"),
}


# ── Prompt helpers (mirrors run_inference_benchmarks.py logic) ─────────────────

def _format_hf_prompt(task_name: str, sample: dict) -> str:
    """Build a prompt string for a single HuggingFace dataset sample."""
    prompt = sample.get('Prompt') or sample.get('prompt') or sample.get('question', '')

    if task_name.startswith("secure_"):
        return prompt  # Use native SECURE prompt directly

    choices = sample.get('answers') or sample.get('choices') or sample.get('options')
    if not choices:
        return prompt

    if task_name == "secbench":
        instruction = "You are given multiple choice questions. Answer with the option letter from the given choices directly."
    else:
        instruction = "You are given multiple choice questions. Answer with the option letter (A, B, C, D) from the given choices directly."

    formatted = instruction + "\n\nQuestion: " + prompt + "\n"
    if isinstance(choices, dict):
        for key in ['A', 'B', 'C', 'D']:
            if key in choices:
                formatted += f"{key}. {choices[key]}\n"
    elif isinstance(choices, list):
        for i, opt in enumerate(choices[:4]):
            formatted += f"{'ABCD'[i]}. {opt}\n"
    return formatted + "Answer:"


def _format_mmlu_prompt(sample: dict, few_shot_prefix: str) -> str:
    choices = sample.get('choices', [])
    question = sample.get('question', '').strip()
    text = question + "\n"
    for i, opt in enumerate(choices[:4]):
        text += f"{'ABCD'[i]}. {opt}\n"
    return few_shot_prefix + text + "Answer:"


def _build_mmlu_prefix() -> str:
    dev = load_dataset("lighteval/mmlu", "computer_security", split="dev")
    header = "The following are multiple choice questions (with answers) about computer security.\n\n"
    parts = [header]
    for s in dev:
        choices = s.get('choices', [])
        ans_idx = s.get('answer')
        q = s.get('question', '').strip()
        text = q + "\n"
        for i, opt in enumerate(choices[:4]):
            text += f"{'ABCD'[i]}. {opt}\n"
        label = 'ABCD'[ans_idx] if isinstance(ans_idx, int) else str(ans_idx)
        parts.append(text + f"Answer: {label}\n\n")
    return "".join(parts)


def _format_jsonl_prompt(sample: dict) -> str:
    return sample.get('prompt') or sample.get('question', '')


def _format_seceval_prompt(q: dict) -> str:
    """Build a calibration prompt for a SecEval question dict."""
    instruction = "Below are multiple-choice questions concerning cybersecurity. Please select the correct answers and respond with the letters ABCD only."
    question = q.get("question", "")
    choices = q.get("choices", [])
    question_text = "Question: " + question + " " + " ".join(choices)
    question_text = question_text.replace("\n", " ")
    return instruction + "\n\n" + question_text


def _format_cybermetric_prompt(sample: dict) -> str:
    """Build a calibration prompt for a CyberMetric sample."""
    system_prompt = "You are a security expert who answers questions."
    question = sample.get("question", "")
    answers = sample.get("answers") or sample.get("choices") or sample.get("options") or {}
    if isinstance(answers, list):
        letters = ["A", "B", "C", "D"]
        answers = {letters[i]: str(opt) for i, opt in enumerate(answers[:4])}
    choices_text = "\n".join(f"{k}. {v}" for k, v in answers.items()) if isinstance(answers, dict) else ""
    prompt = f"{question}\n{choices_text}\nAlways return in this format: 'ANSWER: X'"
    return system_prompt + "\n\n" + prompt


def load_sample_prompts(task_name: str, n_samples: int) -> list:
    """Return up to n_samples prompt strings for calibration."""
    if task_name not in TASK_MAP:
        print(f"  [WARN] Task '{task_name}' not in TASK_MAP — skipping.")
        return []

    src_type, source, subset = TASK_MAP[task_name]

    if src_type == "special" and task_name == "seceval":
        import requests as _req
        try:
            resp = _req.get(source, timeout=30)
            resp.raise_for_status()
            questions = resp.json()[:n_samples]
            return [_format_seceval_prompt(q) for q in questions if q.get("question")]
        except Exception as e:
            print(f"  [WARN] Could not load seceval: {e}")
            return []

    if src_type == "hf":
        if task_name == "mmlu-cs":
            prefix = _build_mmlu_prefix()
            ds = load_dataset(source, subset, split="test")
            ds = ds.select(range(min(n_samples, len(ds))))
            return [_format_mmlu_prompt(s, prefix) for s in ds]
        elif task_name == "cybermetric":
            ds = load_dataset(source, subset, split="test")
            ds = ds.select(range(min(n_samples, len(ds))))
            return [_format_cybermetric_prompt(dict(s)) for s in ds]
        else:
            try:
                ds = load_dataset(source, subset, split="test")
            except Exception:
                ds = load_dataset(source, subset)
                ds = list(ds.values())[0]
            ds = ds.select(range(min(n_samples, len(ds))))
            return [_format_hf_prompt(task_name, s) for s in ds]

    elif src_type == "jsonl":
        ds = load_jsonl_dataset(source)
        ds = ds.select(range(min(n_samples, len(ds))))
        return [_format_jsonl_prompt(dict(s)) for s in ds]

    return []


# ── Token counting helpers ─────────────────────────────────────────────────────

def count_think_tokens(text: str, tokenizer) -> int:
    """Count tokens inside <think>...</think> blocks (for thinking models)."""
    think_text = " ".join(re.findall(r'<think>(.*?)</think>', text, re.DOTALL))
    if not think_text:
        return 0
    return len(tokenizer.encode(think_text, add_special_tokens=False))


def round_up_to_next_power_of_2(n: int, minimum: int = 256) -> int:
    """Round n up to the nearest power of 2, but at least minimum."""
    n = max(n, minimum)
    return int(2 ** math.ceil(math.log2(n)))


def compute_recommended(p99: int, ceiling: int) -> int:
    """Compute recommended max_tokens from p99 with 50% headroom, capped at ceiling."""
    recommended = round_up_to_next_power_of_2(int(p99 * 1.5), minimum=256)
    return min(recommended, ceiling)


# ── Main calibration logic ─────────────────────────────────────────────────────

def calibrate_task(task_name: str, llm, tokenizer, n_samples: int,
                   ceiling: int, is_thinking_model: bool) -> dict:
    """Run calibration for a single task and return statistics dict."""
    print(f"\n  [{task_name}] Loading {n_samples} sample prompts...")
    prompts = load_sample_prompts(task_name, n_samples)
    if not prompts:
        return None

    # Apply chat template to each prompt
    formatted = []
    for p in prompts:
        try:
            msgs = [{"role": "user", "content": p}]
            fp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            formatted.append(fp)
        except Exception:
            formatted.append(p)

    # Run inference with ceiling as max_tokens
    sampling_params = SamplingParams(max_tokens=ceiling, temperature=0.0, top_p=1.0)
    outputs = llm.generate(formatted, sampling_params)

    token_counts = []
    think_counts = []
    truncated = 0

    for out in outputs:
        text = out.outputs[0].text
        n_tokens = len(out.outputs[0].token_ids)
        token_counts.append(n_tokens)

        if is_thinking_model:
            think_counts.append(count_think_tokens(text, tokenizer))

        if n_tokens >= ceiling:
            truncated += 1

    arr = np.array(token_counts)
    think_arr = np.array(think_counts) if think_counts else np.zeros(len(token_counts))

    p99 = int(np.percentile(arr, 99))
    stats = {
        "recommended": compute_recommended(p99, ceiling),
        "mean": int(arr.mean()),
        "p50":  int(np.percentile(arr, 50)),
        "p90":  int(np.percentile(arr, 90)),
        "p95":  int(np.percentile(arr, 95)),
        "p99":  p99,
        "think_p95": int(np.percentile(think_arr, 95)),
        "truncated_pct": round(truncated / len(token_counts), 4),
        "n_samples": len(token_counts),
        "ceiling_used": ceiling,
    }
    print(f"  [{task_name}] mean={stats['mean']} p95={stats['p95']} p99={p99} "
          f"truncated={truncated}/{len(token_counts)} → recommended={stats['recommended']}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate per-task max_tokens for a given model via vLLM."
    )
    parser.add_argument("--model_path", required=True,
                        help="HuggingFace model ID or local path to model directory")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of samples per task (default: 10)")
    parser.add_argument("--ceiling", type=int, default=2048,
                        help="Max tokens ceiling used during calibration (default: 2048)")
    parser.add_argument("--gpu_mem", type=float, default=0.90,
                        help="vLLM GPU memory utilization (default: 0.90)")
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="Maximum model context length for vLLM (default: 8192)")
    parser.add_argument("--output", required=True,
                        help="Path to write calibration JSON output")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                        help="Tasks to calibrate (default: all supported tasks)")
    parser.add_argument("--is_thinking_model", action="store_true",
                        help="Set for models that produce <think>...</think> chains (e.g. Qwen3, DeepSeek-R1)")
    args = parser.parse_args()

    if not HAS_VLLM:
        raise RuntimeError("vLLM is required for calibration. Install with: pip install vllm")

    import torch
    print(f"\n{'='*70}")
    print(f"Token Calibration")
    print(f"Model:    {args.model_path}")
    print(f"Samples:  {args.n_samples} per task")
    print(f"Ceiling:  {args.ceiling} tokens")
    print(f"Tasks:    {args.tasks}")
    print(f"Output:   {args.output}")
    print(f"{'='*70}\n")

    # Initialise vLLM
    print("Initialising vLLM...")
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_mem,
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    # Access tokenizer from the vLLM engine for chat template support
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Run calibration per task
    task_stats = {}
    task_recommended = {}

    for task_name in args.tasks:
        try:
            stats = calibrate_task(
                task_name, llm, tokenizer,
                args.n_samples, args.ceiling, args.is_thinking_model
            )
            if stats:
                task_stats[task_name] = stats
                task_recommended[task_name] = stats["recommended"]
        except Exception as e:
            print(f"  [ERROR] {task_name}: {e}")
            continue

    # Build output JSON
    output = {
        "model": args.model_path,
        "is_thinking_model": args.is_thinking_model,
        "calibrated_at": datetime.now().isoformat(),
        "n_samples": args.n_samples,
        "ceiling": args.ceiling,
        "tasks": task_stats,
        # Flat recommended values for easy lookup by run_inference_benchmarks.py
        **task_recommended,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Calibration complete. Results written to: {args.output}")
    print(f"{'='*70}\n")
    print("Per-task recommended max_tokens:")
    for task, val in task_recommended.items():
        print(f"  {task:30s}: {val}")


if __name__ == "__main__":
    main()
