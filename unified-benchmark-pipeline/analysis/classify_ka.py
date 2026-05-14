"""Run K-vs-A classification with one model on the K/A sample bank.

Two backends:
  --backend api     : Azure GPT-5.4 (default; reuses analysis.lib.verify path)
  --backend vllm    : local vLLM (for the open-weight models)

Cache: per-(model, task, idx) JSON at
       analysis/reports/coverage/verdicts/<model_label>/<task>_<idx>.json

Resumable: re-running skips cached items.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from analysis.lib.ka_classifier import SYS, build_user_prompt, parse_verdict


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ENV_PATH = PROJECT_ROOT / ".env"
COVERAGE_DIR = HERE / "reports" / "coverage"
BANK_PATH = COVERAGE_DIR / "sample_bank.jsonl"


def load_env():
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def cache_path(model_label: str, task: str, idx: str) -> Path:
    return COVERAGE_DIR / "verdicts" / model_label / f"{task}_{idx}.json"


def load_bank():
    return [json.loads(l) for l in BANK_PATH.open() if l.strip()]


def save_verdict(model_label: str, rec: dict, raw_text: str, parsed: dict | None):
    cp = cache_path(model_label, rec["task"], str(rec["index"]))
    cp.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "task": rec["task"],
        "index": rec["index"],
        "model_label": model_label,
        "class": (parsed or {}).get("class"),
        "confidence": (parsed or {}).get("confidence"),
        "rationale": (parsed or {}).get("rationale", ""),
        "raw_text": raw_text,
    }
    cp.write_text(json.dumps(out))
    return out


# ----------------------- Azure backend -----------------------

def run_api(model_label: str, bank: list, time_budget_s: float, limit: int):
    from openai import AzureOpenAI
    load_env()
    key = os.environ.get("AZURE_API_KEY")
    if not key:
        sys.exit("AZURE_API_KEY not set")
    client = AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/",
        api_key=key,
    )
    new = 0
    t0 = time.time()
    for rec in bank:
        if cache_path(model_label, rec["task"], str(rec["index"])).exists():
            continue
        if limit and new >= limit:
            break
        if (time.time() - t0) > time_budget_s:
            print(f"time budget reached; processed {new}")
            break
        try:
            resp = client.chat.completions.create(
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": build_user_prompt(rec["question"])},
                ],
                max_completion_tokens=400,
                temperature=0.0,
            )
            text = (resp.choices[0].message.content or "").strip()
            parsed = parse_verdict(text)
        except Exception as e:
            text = f"[error] {type(e).__name__}: {e}"
            parsed = None
        save_verdict(model_label, rec, text, parsed)
        new += 1
    print(f"new this run: {new}")


# ----------------------- vLLM backend -----------------------

def run_vllm(model_label: str, model_path: str, bank: list, batch_size: int, limit: int):
    from vllm import LLM, SamplingParams
    todo = [r for r in bank if not cache_path(model_label, r["task"], str(r["index"])).exists()]
    if limit:
        todo = todo[:limit]
    if not todo:
        print("nothing to do (all cached)")
        return
    print(f"loading vLLM model: {model_path}")
    import os as _os
    dtype = _os.environ.get("VLLM_DTYPE", "half")
    tp = int(_os.environ.get("VLLM_TP", "1"))
    llm = LLM(
        model=model_path,
        dtype=dtype,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=400,
    )
    prompts = []
    for r in todo:
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": build_user_prompt(r["question"])},
        ]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    print(f"generating {len(prompts)} prompts in batches of {batch_size}")
    new = 0
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        outs = llm.generate(batch, sp, use_tqdm=False)
        for rec, out in zip(todo[start:start + batch_size], outs):
            text = out.outputs[0].text.strip() if out.outputs else ""
            parsed = parse_verdict(text)
            save_verdict(model_label, rec, text, parsed)
            new += 1
        print(f"  done {new}/{len(todo)}", flush=True)
    print(f"finished: {new}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-label", required=True,
                   help="Output cache directory name (e.g. 'GPT-5.4', 'Qwen3.6-35B-A3B').")
    p.add_argument("--backend", choices=["api", "vllm"], required=True)
    p.add_argument("--model-path", default=None,
                   help="Required when backend=vllm; HF id or local model directory.")
    p.add_argument("--time-budget-s", type=float, default=1800.0,
                   help="API backend only.")
    p.add_argument("--batch-size", type=int, default=64, help="vLLM batch size.")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    bank = load_bank()
    print(f"bank size: {len(bank)}")
    print(f"already cached for {args.model_label}: "
          f"{sum(1 for r in bank if cache_path(args.model_label, r['task'], str(r['index'])).exists())}")

    if args.backend == "api":
        run_api(args.model_label, bank, args.time_budget_s, args.limit)
    else:
        if not args.model_path:
            sys.exit("--model-path required for vllm backend")
        run_vllm(args.model_label, args.model_path, bank, args.batch_size, args.limit)


if __name__ == "__main__":
    main()
