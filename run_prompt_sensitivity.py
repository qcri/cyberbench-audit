#!/usr/bin/env python3
"""
Prompt sensitivity study across cybersecurity benchmarks.

For each task, collects 100 seeded samples and runs three prompt modes:
  zero_shot  — original benchmark prompt, no modification
  few_shot   — 2 real samples from the dataset (seeded, excluded from test pool)
               prepended as Q+A examples before each test question
  cot        — COT_SYSTEM_SUFFIX added to system prompt only; no examples in prompt;
               enable_thinking=True so native thinking (Qwen3, Gemma4) runs freely.
               token budget raised to COT_MAX_NEW_TOKENS for all tasks

Outputs per task/mode:
  {output_dir}/{task}_{mode}_responses.jsonl   — same schema as run_inference_benchmarks.py
  {output_dir}/fewshot_examples_{task}.json    — which samples were used as few-shot examples

Supported tasks (24 total):
  CTI-Bench:   mcq, rcm, rcm_2021, vsp, ate, cti_taa
  AthenaBench: ckt, athena_ate, athena_rcm, rms, athena_vsp, taa
  SECURE:      secure_maet, secure_cwet, secure_kcv
  Others:      seceval, cybermetric, secbench, mmlu-cs
  RedSage:     redsage_frameworks, redsage_generals, redsage_skills, redsage_cli, redsage_kali

Few-shot design is benchmark-specific by intent:
  - Most tasks: n_shot real samples drawn from dataset (seeded, excluded from test pool)
  - SecEval:   fixed official few-shot examples from the benchmark's eval.py
  - MMLU-CS:   always 5-shot from the dev split (faithful to original benchmark)

Note: enable_thinking only affects local HF and vLLM inference via apply_chat_template.
For API runs (Azure Claude, Azure OpenAI, OpenAI-compat) the flag has no effect —
those endpoints do not expose a thinking toggle at the prompt level.
"""

import os
import json
import random
import argparse
import re
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset, Dataset

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = None
    HAS_TRANSFORMERS = False

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# =============================================================================
# Configuration constants
# =============================================================================

N_SHOT = 2       # number of few-shot examples drawn from each dataset
N_TEST = 100     # test samples per task/mode
COT_MAX_NEW_TOKENS = 8192  # generous budget — Qwen3 CoT hits ~6000 tokens on VSP/TAA tasks

# Per-task CoT overrides — tasks whose CoT responses exceed COT_MAX_NEW_TOKENS
COT_TASK_MAX_NEW_TOKENS = {
    "ate":         16384,  # Llama-Primus CoT ~6700, gpt-oss max=8222
    "ckt":         16384,  # gpt-oss max=8930
    "athena_ate":  16384,  # gpt-oss 8054, Fanar 9796
    "rms":         16384,  # gpt-oss 8173, Fanar 8418
    "athena_vsp":  16384,  # gpt-oss max=9455
    "redsage_cli": 16384,  # gpt-oss max=7769
    "rcm_2021":    16384,  # Fanar max=8993
    "seceval":     16384,  # Fanar max=8947
}

# Original token budgets (used for zero_shot and few_shot)
TASK_MAX_NEW_TOKENS = {
    "mcq":                2048,
    "rcm":                2048,
    "rcm_2021":           2048,
    "vsp":                2048,
    "ate":                8192,   # Llama-Primus zero_shot hits ~6800 tokens
    "cti_taa":            4096,   # Llama33 zero_shot p95=3288 exceeds original 2048
    "ckt":                2048,
    "athena_ate":         2048,
    "athena_rcm":         2048,
    "rms":                2048,
    "athena_vsp":         2048,
    "taa":                2048,
    "secure_maet":        1024,
    "secure_cwet":        1024,
    "secure_kcv":         1024,
    "seceval":            2048,  # original 5 only works with tight instruction; zero/few_shot need room
    "cybermetric":        1024,
    "secbench":           16,
    "redsage_frameworks": 100,
    "redsage_generals":   100,
    "redsage_skills":     100,
    "redsage_cli":        100,
    "redsage_kali":       100,
    "mmlu-cs":            4096,  # original 5 assumes brief answer; zero/few_shot generate longer responses
}

# RedSage tasks use a newline stop sequence in zero/few_shot; removed for cot
REDSAGE_TASKS = {
    "redsage_frameworks", "redsage_generals", "redsage_skills",
    "redsage_cli", "redsage_kali",
}

# System prompts per task (None = no system prompt, faithful to original)
TASK_SYSTEM_PROMPTS = {
    "mcq":        "You are a cybersecurity expert specializing in cyberthreat intelligence.",
    "rcm":        "You are a cybersecurity expert specializing in cyberthreat intelligence.",
    "rcm_2021":   "You are a cybersecurity expert specializing in cyberthreat intelligence.",
    "vsp":        "You are a cybersecurity expert specializing in cyberthreat intelligence.",
    "ate":        "You are a cybersecurity expert specializing in cyberthreat intelligence.",
    "cti_taa":    "You are a cybersecurity expert specializing in cyberthreat intelligence.",
    "cybermetric": "You are a security expert who answers questions.",
}

# Single general CoT system suffix — appended to the existing system prompt for cot mode.
# Task-agnostic: invokes structured reasoning via <scratchpad> tags without prescribing domain content.
# Applied identically across all 24 tasks.
COT_SYSTEM_SUFFIX = """

You are an expert problem-solving assistant with strong analytical skills.

Before responding, always place your full reasoning inside <scratchpad></scratchpad> tags.

Inside <scratchpad>, follow this approach:
1. Break down the question into its component parts
2. Clearly state your assumptions and what information is given
3. Develop a structured reasoning path step by step
4. Consider multiple interpretations or answer candidates
5. Evaluate evidence and rule out alternatives explicitly
6. Draw a well-justified conclusion

When reasoning:
- Use explicit step-by-step logic
- Identify key variables and constraints
- Explore alternative scenarios before committing
- Highlight any uncertainty and resolve it
- For quantitative problems: work through each component systematically and verify consistency
- For qualitative problems: assess how factors interact and weigh competing explanations

After </scratchpad>, provide only your final answer in the exact format the question asks for."""

# =============================================================================
# Helpers copied from run_inference_benchmarks.py (unchanged)
# =============================================================================

MMLU_CHOICES = ["A", "B", "C", "D"]

def format_mmlu_subject(subject: str) -> str:
    return " ".join([""] + subject.split("_"))

def mmlu_answer_letter(answer_val):
    if isinstance(answer_val, int):
        return MMLU_CHOICES[answer_val]
    return str(answer_val).strip()

def format_mmlu_example(sample: dict, include_answer: bool = True) -> str:
    prompt = sample["question"]
    for j, choice in enumerate(sample["choices"]):
        prompt += "\n{}. {}".format(MMLU_CHOICES[j], choice)
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(mmlu_answer_letter(sample["answer"]))
    return prompt

def gen_mmlu_prompt(dev_rows: list, subject: str, k: int = -1) -> str:
    prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
        format_mmlu_subject(subject)
    )
    if k == -1:
        k = len(dev_rows)
    for i in range(k):
        prompt += format_mmlu_example(dev_rows[i], include_answer=True)
    return prompt

REDSAGE_CHOICES = ["A", "B", "C", "D"]

def build_redsage_prompt(sample: dict, include_context: bool = False) -> str:
    content  = sample.get("content", "")
    question = sample["question"]
    answers  = sample["answers"]
    query_parts = []
    if include_context and content:
        query_parts.append("Context: " + content)
    query_parts.append("Question: " + question)
    choices_str = "\n".join(f"{l}. {answers[l]}" for l in REDSAGE_CHOICES)
    instruction = "You are given multiple choice questions. Answer with the option letter (A, B, C, D) from the given choices directly."
    return instruction + "\n" + "\n\n".join(query_parts) + "\n" + choices_str + "\nAnswer:"

def redsage_answer_letter(answer_val):
    return str(answer_val).strip().upper()

def apply_stop_sequence(text: str, stop_sequence=None) -> str:
    if not stop_sequence:
        return text
    earliest = None
    for stop in stop_sequence:
        pos = text.find(stop)
        if pos != -1 and (earliest is None or pos < earliest):
            earliest = pos
    return text[:earliest] if earliest is not None else text

def load_json_dataset(source: str):
    if source.startswith("http://") or source.startswith("https://"):
        r = requests.get(source, timeout=30)
        r.raise_for_status()
        return r.json()
    with open(source, "r", encoding="utf-8") as f:
        return json.load(f)

def load_concatenated_json_objects(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        r = requests.get(source, timeout=30)
        r.raise_for_status()
        text = r.text
    else:
        with open(source, "r", encoding="utf-8") as f:
            text = f.read()
    decoder = json.JSONDecoder()
    idx, data = 0, []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        data.append(obj)
        idx = end
    return data

def load_tsv_dataset(source: str) -> Dataset:
    if source.startswith("http://") or source.startswith("https://"):
        r = requests.get(source, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), sep="\t", encoding="utf-8")
    else:
        df = pd.read_csv(source, sep="\t", encoding="utf-8")
    return Dataset.from_list(df.replace({np.nan: None}).to_dict(orient="records"))

def load_jsonl_dataset(source: str) -> Dataset:
    if source.startswith("http://") or source.startswith("https://"):
        r = requests.get(source, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
    else:
        with open(source, "r", encoding="utf-8") as f:
            lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if line:
            obj = json.loads(line)
            data.append({k: ("" if v is None else str(v)) for k, v in obj.items()})
    return Dataset.from_list(data)

def load_model_and_tokenizer(model_path: str, base_model: str = None, is_base: bool = False):
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model if base_model else model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model if base_model else model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if base_model and not is_base and HAS_PEFT:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        model = base
    model.eval()
    print(f"Model loaded on: {next(model.parameters()).device}")
    return model, tokenizer

def chat_completion_api(endpoint, model_name, prompt=None, api_key="",
                        max_tokens=1024, temperature=0.0, top_p=1.0,
                        seed=None, retries=3, system_prompt=None, messages=None) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
    payload = {"model": model_name, "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
    if seed is not None:
        payload["seed"] = seed
    for attempt in range(retries):
        try:
            r = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                print(f"API call failed (attempt {attempt+1}/{retries}): {e}")
            else:
                return f"ERROR: {e}"

def chat_completion_azure_openai(endpoint, model_name, prompt=None, api_key="",
                                  max_tokens=1024, temperature=0.0, top_p=1.0,
                                  seed=None, retries=3, system_prompt=None, messages=None) -> str:
    import urllib.request as _req, urllib.error as _err
    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
    api_key = api_key.encode("ascii", errors="ignore").decode("ascii").strip()
    payload = {"model": model_name, "input": messages,
               "max_output_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(retries):
        try:
            req = _req.Request(endpoint, data=body,
                               headers={"Content-Type": "application/json", "api-key": api_key},
                               method="POST")
            with _req.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["output"][0]["content"][0]["text"]
        except _err.HTTPError as e:
            body_txt = e.read().decode("utf-8", errors="replace")
            print(f"Azure HTTP {e.code} (attempt {attempt+1}/{retries}): {body_txt[:300]}")
            if attempt >= retries - 1:
                return f"ERROR: HTTP {e.code}: {body_txt[:200]}"
        except Exception as e:
            if attempt >= retries - 1:
                return f"ERROR: {e}"

def chat_completion_azure_claude(endpoint, model_name, prompt=None, api_key="",
                                  max_tokens=1024, temperature=0.0, top_p=1.0,
                                  seed=None, retries=3, system_prompt=None, messages=None) -> str:
    headers = {"Content-Type": "application/json", "x-api-key": api_key,
               "anthropic-version": "2023-06-01"}
    if messages is None:
        messages = []
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
    user_messages = [m for m in messages if m.get("role") != "system"]
    resolved_system = system_prompt or next(
        (m["content"] for m in messages if m.get("role") == "system"), None)
    payload = {"model": model_name, "messages": user_messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if resolved_system:
        payload["system"] = resolved_system
    for attempt in range(retries):
        try:
            r = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["content"][0]["text"]
        except Exception as e:
            if attempt >= retries - 1:
                return f"ERROR: {e}"

def generate_response(model, tokenizer, prompt=None, max_new_tokens=1024,
                      use_api=False, use_vllm=False, vllm_model=None,
                      api_endpoint=None, api_model=None, api_key="",
                      api_type="openai_compat", batch_size=None,
                      system_prompt=None, messages=None,
                      temperature=0.0, top_p=1.0, seed=None,
                      enable_thinking=True, **kwargs) -> str:
    _dispatch = {"openai_compat": chat_completion_api,
                 "azure_openai":  chat_completion_azure_openai,
                 "azure_claude":  chat_completion_azure_claude}
    _call = _dispatch.get(api_type, chat_completion_api)

    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})

    if use_api:
        return _call(api_endpoint, api_model, api_key=api_key,
                     max_tokens=max_new_tokens, temperature=temperature,
                     top_p=top_p, seed=seed, messages=messages)

    if use_vllm and vllm_model:
        sp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        if tokenizer is not None:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n".join(m.get("content", "") for m in messages)
        out = vllm_model.generate([text], sp)
        return out[0].outputs[0].text.strip()

    try:
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    if isinstance(inputs, dict):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
    else:
        inputs = inputs.to(model.device)
        input_len = inputs.shape[1]
        inputs = {"input_ids": inputs}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

def initialize_vllm(model_path, base_model=None, gpu_memory_utilization=0.9,
                    max_model_len=None, enforce_eager=False):
    """Initialize vLLM engine and load tokenizer separately for chat template formatting.

    The tokenizer is needed so _apply_chat_template can call tokenizer.apply_chat_template()
    with the correct instruct template (including enable_thinking for Qwen3 etc.).
    Without it, prompt formatting falls back to plain text concatenation.
    """
    import os as _os
    cuda_visible = _os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        n_gpus = len([x for x in cuda_visible.split(',') if x.strip()])
    else:
        n_gpus = int(_os.environ.get('SLURM_GPUS_ON_NODE', '1'))
    print(f"Initializing vLLM: {model_path} (tensor_parallel_size={n_gpus})")
    kwargs = {"gpu_memory_utilization": gpu_memory_utilization,
              "trust_remote_code": True, "enforce_eager": enforce_eager,
              "tensor_parallel_size": n_gpus,
              "disable_log_stats": True,
              "disable_custom_all_reduce": True}
    if max_model_len:
        kwargs["max_model_len"] = max_model_len
    llm = LLM(model=model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model if base_model else model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return llm, tokenizer

def _apply_chat_template(tokenizer, msgs, enable_thinking=True):
    """Apply chat template with thinking control.

    Different models use different parameters:
      - Qwen3, Gemma4: enable_thinking=True/False
      - Fanar: no_thinking=True suppresses thinking (True = disable, inverted logic)
      - Others: no thinking parameter; falls back to plain template

    enable_thinking=True  → model should think (CoT mode)
    enable_thinking=False → model should NOT think (zero/few_shot mode)
    """
    if tokenizer is None:
        sys = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        return (sys + "\n" if sys else "") + user

    # Fanar uses no_thinking=True to suppress thinking (inverted from enable_thinking)
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            no_thinking=not enable_thinking,
        )
    except TypeError:
        pass

    # Qwen3, Gemma4 use enable_thinking
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        pass

    # No thinking parameter supported — use plain template
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _vllm_batch_generate(vllm_model, tokenizer, prompt_tuples, max_new_tokens,
                          temperature=0.0, top_p=1.0, seed=None, stop=None,
                          enable_thinking=True):
    formatted = []
    for user_prompt, system_prompt in prompt_tuples:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_prompt})
        formatted.append(_apply_chat_template(tokenizer, msgs, enable_thinking))
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature,
                        top_p=top_p, stop=stop or [])
    outputs = vllm_model.generate(formatted, sp)
    return [o.outputs[0].text.strip() for o in outputs]

def _vllm_batch_generate_messages(vllm_model, tokenizer, all_messages, max_new_tokens,
                                   temperature=0.0, top_p=1.0, seed=None, stop=None,
                                   enable_thinking=True):
    formatted = []
    for msgs in all_messages:
        formatted.append(_apply_chat_template(tokenizer, msgs, enable_thinking))
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature,
                        top_p=top_p, stop=stop or [])
    outputs = vllm_model.generate(formatted, sp)
    return [o.outputs[0].text.strip() for o in outputs]

# =============================================================================
# Sample selection
# =============================================================================

def split_fewshot_and_test(samples, n_shot=N_SHOT, n_test=N_TEST, seed=42):
    """Seed-reproducible split: first n_shot items become few-shot examples,
    remaining pool is randomly sampled for the n_test test items.
    Returns (fewshot_samples, test_samples, fewshot_original_indices, test_original_indices).
    """
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    fewshot_indices = indices[:n_shot]
    pool_indices    = indices[n_shot:]
    test_indices    = rng.sample(pool_indices, min(n_test, len(pool_indices)))
    return (
        [samples[i] for i in fewshot_indices],
        [samples[i] for i in test_indices],
        fewshot_indices,
        test_indices,
    )

def save_fewshot_metadata(output_dir, task_name, fewshot_samples, fewshot_indices, seed):
    record = {
        "task":    task_name,
        "seed":    seed,
        "n_shot":  len(fewshot_samples),
        "examples": [
            {"original_index": int(idx),
             "prompt":         s["prompt"],
             "ground_truth":   s["ground_truth"]}
            for idx, s in zip(fewshot_indices, fewshot_samples)
        ],
    }
    path = os.path.join(output_dir, f"fewshot_examples_{task_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"  [fewshot] Examples saved → {path}")

# =============================================================================
# Prompt mode application
# =============================================================================

def build_fewshot_prefix(fewshot_samples):
    """Build a text block from real dataset examples (used in few_shot mode)."""
    lines = []
    for s in fewshot_samples:
        lines.append(s["prompt"].rstrip())
        lines.append(f"Answer: {s['ground_truth']}\n")
    lines.append("Now answer the following:\n")
    return "\n".join(lines)

def apply_prompt_mode(mode, task_name, prompt, system_prompt, fewshot_prefix):
    """Return (modified_prompt, modified_system) for the given mode.

    Native thinking (Qwen3 etc.) is suppressed for zero_shot/few_shot via
    enable_thinking=False in apply_chat_template — not through system prompt flags.
    For cot mode, enable_thinking=True lets native thinking run alongside COT_SYSTEM_SUFFIX.
    """
    if mode == "zero_shot":
        return prompt, system_prompt

    elif mode == "few_shot":
        return fewshot_prefix + prompt, system_prompt

    else:  # cot
        modified_system = ((system_prompt or "") + COT_SYSTEM_SUFFIX).strip()
        return prompt, modified_system or None

# =============================================================================
# Dataset loaders — each returns list of {"prompt", "ground_truth", "original_fields"}
# =============================================================================

def load_ctibench_samples(tsv_url):
    ds = load_tsv_dataset(tsv_url)
    return [{"prompt": s["Prompt"],
             "ground_truth": str(s.get("GT") or "").strip(),
             "original_fields": dict(s)}
            for s in ds if s.get("Prompt")]

def load_athenabench_samples(jsonl_url):
    ds = load_jsonl_dataset(jsonl_url)
    return [{"prompt": s["prompt"],
             "ground_truth": str(s.get("answer") or "").strip(),
             "original_fields": dict(s)}
            for s in ds if s.get("prompt")]

def load_secure_samples(tsv_url):
    ds = load_tsv_dataset(tsv_url)
    return [{"prompt": s["Prompt"],
             "ground_truth": str(s.get("Correct Answer") or "").strip(),
             "original_fields": dict(s)}
            for s in ds if s.get("Prompt")]

def load_seceval_samples():
    url = "https://huggingface.co/datasets/XuanwuAI/SecEval/resolve/main/questions.json"
    questions = requests.get(url, timeout=30).json()
    results = []
    for q in questions:
        if not q.get("question") or not q.get("choices"):
            continue
        qt = ("Question: " + q["question"] + " ".join(q["choices"])).replace("\n", " ")
        results.append({"prompt": qt,
                        "ground_truth": q.get("answer", ""),
                        "original_fields": q})
    return results

def load_cybermetric_samples(url=None):
    if url is None:
        url = "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-500-v1.json"
    data = load_json_dataset(url)

    def _build(s):
        opts = ", ".join(f"{k}) {v}" for k, v in s["answers"].items())
        return (f"Question: {s['question']}\nOptions: {opts}\n\n"
                f"Choose the correct answer (A, B, C, or D) only. "
                f"Always return in this format: 'ANSWER: X'")

    return [{"prompt": _build(s),
             "ground_truth": str(s["solution"]).strip(),
             "original_fields": s}
            for s in data["questions"]]

def load_secbench_samples(url):
    raw = load_concatenated_json_objects(url)
    samples = [s for s in raw
               if s.get("language") == "English"
               and s.get("question") and s.get("answers") and s.get("label")]

    def _build(s):
        p = ("Answer the following multiple-choice cybersecurity question. "
             "Select the correct option letter(s) from A, B, C, and D. "
             "Return only the letter(s), with no explanation.\n\n")
        p += str(s["question"]).strip() + "\n"
        for i, a in enumerate(s["answers"][:4]):
            p += f"{chr(65+i)}. {a}\n"
        return p + "Answer:"

    return [{"prompt": _build(s),
             "ground_truth": str(s.get("label") or "").strip(),
             "original_fields": s}
            for s in samples]

def load_redsage_samples(dataset_name, subset_name):
    ds = load_dataset(dataset_name, subset_name, split="test")
    return [{"prompt": build_redsage_prompt(dict(s), include_context=False),
             "ground_truth": redsage_answer_letter(s["solution"]),
             "original_fields": dict(s)}
            for s in ds]

# =============================================================================
# Unified collector
# =============================================================================

def collect_with_mode(task_name, test_samples, mode, output_file,
                      fewshot_prefix, system_prompt, max_new_tokens, stop_seq,
                      model, tokenizer, **api_kwargs):
    """Run inference for one task/mode, write JSONL, return sample count."""
    use_vllm   = api_kwargs.get("use_vllm", False)
    vllm_model = api_kwargs.get("vllm_model")

    prompt_pairs = [
        apply_prompt_mode(mode, task_name, s["prompt"], system_prompt, fewshot_prefix)
        for s in test_samples
    ]

    thinking = (mode == "cot")
    if use_vllm and vllm_model:
        print(f"  [vLLM batch] {len(prompt_pairs)} prompts")
        responses = _vllm_batch_generate(
            vllm_model, tokenizer, prompt_pairs,
            max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0,
            seed=42, stop=stop_seq if stop_seq else None,
            enable_thinking=thinking,
        )
    else:
        responses = []
        for prompt, sys in tqdm(prompt_pairs, desc=f"{task_name}/{mode}"):
            responses.append(generate_response(
                model, tokenizer, prompt,
                enable_thinking=thinking,
                system_prompt=sys,
                max_new_tokens=max_new_tokens,
                temperature=0.0, top_p=1.0, seed=42,
                **api_kwargs,
            ))

    results = []
    for idx, (s, (prompt, sys), response) in enumerate(
            zip(test_samples, prompt_pairs, responses)):
        # For RedSage zero/few_shot apply the stop sequence post-generation
        if stop_seq and mode != "cot":
            response = apply_stop_sequence(response, stop_seq)
        results.append({
            "task":          task_name,
            "index":         idx,
            "prompt":        prompt,
            "ground_truth":  s["ground_truth"],
            "model_response": response,
            "metadata": {
                "prompt_mode":   mode,
                "system_prompt": sys,
                "generation_params": {
                    "temperature":               0.0,
                    "top_p":                     1.0,
                    "seed":                      42,
                    "max_new_tokens":            max_new_tokens,
                    "original_benchmark_budget": TASK_MAX_NEW_TOKENS.get(task_name),
                    "budget_overridden_for_cot": mode == "cot",
                    "stop_sequence":             stop_seq,
                    "stop_sequence_removed_for_cot":
                        mode == "cot" and task_name in REDSAGE_TASKS,
                },
                "original_fields": s.get("original_fields", {}),
            },
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(results)} responses → {output_file}")
    return len(results)


def collect_seceval_with_mode(test_samples, mode, output_file, fewshot_samples,
                               max_new_tokens, model, tokenizer, **api_kwargs):
    """SecEval uses chat messages format; handled separately.

    SecEval provides its own official few-shot examples.
    zero_shot : no examples
    few_shot  : official SecEval examples (faithful to the benchmark)
    cot       : COT_SYSTEM_SUFFIX only — no examples, separate from few_shot
    """
    base_system = (
        "Below are multiple-choice questions concerning cybersecurity. "
        "Please select the correct answers and respond with the letters ABCD only."
    )

    # Official SecEval few-shot examples from the benchmark's eval code
    official_shots = [
        {"role": "user",      "content": "Question: Which mitigation prevent stack overflow bug? A: Stack Canary. B: ALSR. C: CFI. D: Code Signing."},
        {"role": "assistant", "content": "Answer: ABC"},
        {"role": "user",      "content": "Question: You are reviewing a piece of code to identify potential vulnerabilities related to NULL pointer dereferences. Which of the following strategies should you consider implementing to mitigate the risk? Choose all that apply. A: Use a programming language that inherently manages pointers and prevents direct memory access. B: Apply input validation to all variables and data stores that receive data from external sources to ensure they are initialized to expected values. C: Explicitly initialize all variables and other data stores, either during declaration or just before the first usage. D: Implement a custom memory management system that automatically assigns a unique identifier to each pointer."},
        {"role": "assistant", "content": "Answer: ABC"},
        {"role": "user",      "content": "Question: Analyze the code snippet provided. Which of the following practices should be implemented in a multithreaded application to prevent NULL pointer dereference vulnerabilities? Select all that apply. A: Ensure that proper locking APIs are used to lock the pointer before the if statement and unlock when it has finished to prevent race conditions. B: Check the return value of all functions that return a value, including network-related functions, and verify that it is non-null before using it. C: Use automated static analysis tools that target this type of weakness, understanding that while not perfect, they can still be effective. D: Verify that a non-nil response is present before deferring response.Body.Close() to handle cases where the Do method returns an error."},
        {"role": "assistant", "content": "Answer: ABCD"},
    ]

    if mode == "cot":
        sys_instr = (base_system + COT_SYSTEM_SUFFIX).strip()
        shots = []
    elif mode == "few_shot":
        sys_instr = base_system
        shots = official_shots
    else:  # zero_shot
        sys_instr = base_system
        shots = []

    def _build_msgs(q_text):
        return [{"role": "system", "content": sys_instr}] + shots + [{"role": "user", "content": q_text}]

    use_vllm   = api_kwargs.get("use_vllm", False)
    vllm_model = api_kwargs.get("vllm_model")

    thinking = (mode == "cot")
    if use_vllm and vllm_model:
        batch_responses = _vllm_batch_generate_messages(
            vllm_model, tokenizer,
            [_build_msgs(s["prompt"]) for s in test_samples],
            max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, seed=42,
            enable_thinking=thinking,
        )
    else:
        batch_responses = None

    results = []
    iter_src = (zip(test_samples, batch_responses) if batch_responses is not None
                else ((s, None) for s in tqdm(test_samples, desc=f"seceval/{mode}")))

    for idx, (s, pre) in enumerate(iter_src):
        msgs = _build_msgs(s["prompt"])
        response = pre if pre is not None else generate_response(
            model, tokenizer, messages=msgs,
            enable_thinking=thinking,
            max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, seed=42, **api_kwargs)
        results.append({
            "task":          "seceval",
            "index":         idx,
            "prompt":        msgs,
            "ground_truth":  s["ground_truth"],
            "model_response": response,
            "metadata": {
                "prompt_mode":   mode,
                "generation_params": {
                    "temperature":               0.0,
                    "top_p":                     1.0,
                    "seed":                      42,
                    "max_new_tokens":            max_new_tokens,
                    "original_benchmark_budget": TASK_MAX_NEW_TOKENS["seceval"],
                    "budget_overridden_for_cot": mode == "cot",
                },
                "original_fields": s.get("original_fields", {}),
            },
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(results)} responses → {output_file}")
    return len(results)


def collect_mmlu_with_mode(dev_rows, test_samples, mode, output_file,
                            fewshot_dev_rows, subset_name,
                            max_new_tokens, model, tokenizer, **api_kwargs):
    """MMLU uses its dev split for few-shot; handled separately."""
    use_vllm   = api_kwargs.get("use_vllm", False)
    vllm_model = api_kwargs.get("vllm_model")

    # cot mode attaches COT_SYSTEM_SUFFIX as system prompt; zero/few_shot have no system prompt
    cot_system = COT_SYSTEM_SUFFIX.strip() if mode == "cot" else None

    def _build(s):
        question_only = format_mmlu_example(s, include_answer=False)
        if mode == "few_shot":
            header   = "The following are multiple choice questions (with answers) about{}.\n\n".format(
                format_mmlu_subject(subset_name))
            examples = "".join(format_mmlu_example(r, include_answer=True) for r in fewshot_dev_rows)
            return header + examples + question_only
        else:  # zero_shot or cot — no examples in the prompt
            return question_only

    prompts = [_build(s) for s in test_samples]

    thinking = (mode == "cot")
    if use_vllm and vllm_model:
        batch_responses = _vllm_batch_generate(
            vllm_model, tokenizer, [(p, cot_system) for p in prompts],
            max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, seed=42,
            enable_thinking=thinking,
        )
    else:
        batch_responses = None

    results = []
    iter_src = (zip(test_samples, prompts, batch_responses) if batch_responses is not None
                else ((s, _build(s), None) for s in tqdm(test_samples, desc=f"mmlu-cs/{mode}")))

    for idx, (s, prompt, pre) in enumerate(iter_src):
        response = pre if pre is not None else generate_response(
            model, tokenizer, prompt, system_prompt=cot_system,
            max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, seed=42, **api_kwargs)
        results.append({
            "task":          "mmlu-cs",
            "index":         idx,
            "prompt":        prompt,
            "ground_truth":  mmlu_answer_letter(s["answer"]),
            "model_response": response,
            "metadata": {
                "prompt_mode":   mode,
                "generation_params": {
                    "temperature":               0.0,
                    "top_p":                     1.0,
                    "seed":                      42,
                    "max_new_tokens":            max_new_tokens,
                    "original_benchmark_budget": TASK_MAX_NEW_TOKENS["mmlu-cs"],
                    "budget_overridden_for_cot": mode == "cot",
                },
                "original_fields": s,
            },
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(results)} responses → {output_file}")
    return len(results)

# =============================================================================
# Main
# =============================================================================

TASK_MAP = {
    "mcq":                ("ctibench_tsv",  "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-mcq.tsv"),
    "rcm":                ("ctibench_tsv",  "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-rcm.tsv"),
    "rcm_2021":           ("ctibench_tsv",  "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-rcm-2021.tsv"),
    "vsp":                ("ctibench_tsv",  "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-vsp.tsv"),
    "ate":                ("ctibench_tsv",  "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-ate.tsv"),
    "cti_taa":            ("ctibench_tsv",  "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-taa.tsv"),
    "ckt":                ("athenabench",   "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-ckt-3k.jsonl"),
    "athena_ate":         ("athenabench",   "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-ate.jsonl"),
    "athena_rcm":         ("athenabench",   "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-rcm.jsonl"),
    "rms":                ("athenabench",   "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-rms.jsonl"),
    "athena_vsp":         ("athenabench",   "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-vsp.jsonl"),
    "taa":                ("athenabench",   "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-taa.jsonl"),
    "secure_maet":        ("secure_tsv",    "https://raw.githubusercontent.com/aiforsec/SECURE/main/Dataset/SECURE%20-%20MAET.tsv"),
    "secure_cwet":        ("secure_tsv",    "https://raw.githubusercontent.com/aiforsec/SECURE/main/Dataset/SECURE%20-%20CWET.tsv"),
    "secure_kcv":         ("secure_tsv",    "https://raw.githubusercontent.com/aiforsec/SECURE/main/Dataset/SECURE%20-%20KCV.tsv"),
    "secbench":           ("secbench",      "https://raw.githubusercontent.com/secbench-git/SecBench/main/data/MCQs_2730.jsonl"),
    "redsage_frameworks": ("redsage",       ("RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_knowledge_frameworks")),
    "redsage_generals":   ("redsage",       ("RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_knowledge_generals")),
    "redsage_skills":     ("redsage",       ("RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_skills")),
    "redsage_cli":        ("redsage",       ("RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_tools_cli")),
    "redsage_kali":       ("redsage",       ("RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_tools_kali")),
}

ALL_TASKS = list(TASK_MAP.keys()) + ["seceval", "cybermetric", "mmlu-cs"]


def main():
    parser = argparse.ArgumentParser(
        description="Prompt sensitivity study: zero-shot vs few-shot vs CoT")

    # Model / inference
    parser.add_argument("--model_path",   type=str)
    parser.add_argument("--base_model",   type=str, default=None)
    parser.add_argument("--is_base",      action="store_true")
    parser.add_argument("--use_api",      action="store_true")
    parser.add_argument("--api_endpoint", type=str)
    parser.add_argument("--api_model",    type=str)
    parser.add_argument("--api_key",      type=str, default="")
    parser.add_argument("--api_type",     type=str, default="openai_compat",
                        choices=["openai_compat", "azure_openai", "azure_claude"])
    parser.add_argument("--use_vllm",     action="store_true")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--batch_size",   type=int, default=16)

    # Study options
    parser.add_argument("--tasks",        nargs="+", default=ALL_TASKS)
    parser.add_argument("--prompt_modes", nargs="+", default=["zero_shot", "few_shot", "cot"],
                        choices=["zero_shot", "few_shot", "cot"])
    parser.add_argument("--n_test",       type=int, default=N_TEST,
                        help="Test samples per task/mode (default 100)")
    parser.add_argument("--n_shot",       type=int, default=N_SHOT,
                        help="Few-shot examples per task (default 2)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--output_dir",   type=str, default=None)

    args = parser.parse_args()

    if args.use_api and args.use_vllm:
        parser.error("Cannot use both --use_api and --use_vllm.")

    # Load model
    if args.use_api:
        if not args.api_endpoint or not args.api_model:
            parser.error("--api_endpoint and --api_model required with --use_api")
        model = tokenizer = vllm_model = None
    elif args.use_vllm:
        if not args.model_path:
            parser.error("--model_path required for vLLM")
        vllm_model, tokenizer = initialize_vllm(args.model_path, args.base_model,
                                                args.vllm_gpu_memory_utilization,
                                                args.max_model_len, args.enforce_eager)
        model = None  # tokenizer loaded by initialize_vllm above
    else:
        if not args.model_path:
            parser.error("--model_path required (or use --use_api / --use_vllm)")
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model, args.is_base)
        vllm_model = None

    # Output directory
    if args.output_dir is None:
        label = args.api_model if args.use_api else args.model_path
        label = label.rstrip("/").split("/")[-1].replace("/", "-")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"sensitivity_{label}_{ts}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Save run metadata
    run_meta = {
        "model": args.api_model if args.use_api else args.model_path,
        "inference_mode": "api" if args.use_api else ("vllm" if args.use_vllm else "local"),
        "tasks": args.tasks,
        "prompt_modes": args.prompt_modes,
        "n_test": args.n_test,
        "n_shot": args.n_shot,
        "seed": args.seed,
        "cot_max_new_tokens": COT_MAX_NEW_TOKENS,
        "timestamp": datetime.now().isoformat(),
        "cot_system_suffix": COT_SYSTEM_SUFFIX,
    }
    with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    api_kwargs = {
        "use_api":      args.use_api,
        "use_vllm":     args.use_vllm,
        "vllm_model":   vllm_model,
        "batch_size":   args.batch_size,
        "api_endpoint": args.api_endpoint,
        "api_model":    args.api_model,
        "api_key":      args.api_key,
        "api_type":     args.api_type,
    }

    summary = {}

    for task_name in args.tasks:
        print(f"\n{'='*70}")
        print(f"Task: {task_name.upper()}")
        print(f"{'='*70}")

        # ---- Load all samples for this task ----
        try:
            if task_name == "seceval":
                all_samples = load_seceval_samples()
            elif task_name == "cybermetric":
                all_samples = load_cybermetric_samples()
            elif task_name == "mmlu-cs":
                dev_rows, mmlu_test_rows = load_dataset("cais/mmlu", "computer_security", split="dev"), \
                                           load_dataset("cais/mmlu", "computer_security", split="test")
                dev_rows      = [dict(x) for x in dev_rows]
                mmlu_test_rows = [dict(x) for x in mmlu_test_rows]
                all_samples   = None  # handled separately below
            elif task_name in TASK_MAP:
                task_type, source = TASK_MAP[task_name]
                if task_type == "ctibench_tsv":
                    all_samples = load_ctibench_samples(source)
                elif task_type == "athenabench":
                    all_samples = load_athenabench_samples(source)
                elif task_type == "secure_tsv":
                    all_samples = load_secure_samples(source)
                elif task_type == "secbench":
                    all_samples = load_secbench_samples(source)
                elif task_type == "redsage":
                    dataset_name, subset_name = source
                    all_samples = load_redsage_samples(dataset_name, subset_name)
                else:
                    print(f"Unknown task type '{task_type}' for {task_name}, skipping.")
                    continue
            else:
                print(f"Unknown task '{task_name}', skipping.")
                continue
        except Exception as e:
            print(f"ERROR loading {task_name}: {e}")
            summary[task_name] = {"error": str(e)}
            continue

        # ---- MMLU: special split using dev rows for few-shot ----
        if task_name == "mmlu-cs":
            rng = random.Random(args.seed)
            fewshot_dev_rows = dev_rows[:5]  # MMLU uses 5-shot from dev (faithful to original)
            test_indices     = rng.sample(range(len(mmlu_test_rows)), min(args.n_test, len(mmlu_test_rows)))
            test_samples     = [mmlu_test_rows[i] for i in test_indices]

            # Save fewshot metadata for MMLU
            fewshot_meta = {
                "task": "mmlu-cs", "seed": args.seed, "n_shot": len(fewshot_dev_rows),
                "source": "cais/mmlu dev split",
                "examples": [{"prompt": format_mmlu_example(r, include_answer=False),
                               "ground_truth": mmlu_answer_letter(r["answer"])}
                             for r in fewshot_dev_rows],
            }
            with open(os.path.join(args.output_dir, "fewshot_examples_mmlu-cs.json"), "w") as f:
                json.dump(fewshot_meta, f, indent=2, ensure_ascii=False)

            task_summary = {}
            for mode in args.prompt_modes:
                out_file = os.path.join(args.output_dir, f"mmlu-cs_{mode}_responses.jsonl")
                if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                    n = sum(1 for _ in open(out_file))
                    print(f"  [RESUME] {mode}: {n} rows already in {out_file}")
                    task_summary[mode] = {"samples": n, "resumed": True}
                    continue
                if mode == "cot":
                    max_tok = COT_TASK_MAX_NEW_TOKENS.get("mmlu-cs", COT_MAX_NEW_TOKENS)
                else:
                    max_tok = TASK_MAX_NEW_TOKENS["mmlu-cs"]
                try:
                    n = collect_mmlu_with_mode(dev_rows, test_samples, mode, out_file,
                                               fewshot_dev_rows, "computer_security",
                                               max_tok, model, tokenizer, **api_kwargs)
                    task_summary[mode] = {"samples": n, "output_file": out_file}
                except Exception as e:
                    print(f"  ERROR {mode}: {e}")
                    task_summary[mode] = {"error": str(e)}
            summary[task_name] = task_summary
            continue

        # ---- All other tasks: seeded split ----
        fewshot_samples, test_samples, fewshot_indices, _ = split_fewshot_and_test(
            all_samples, n_shot=args.n_shot, n_test=args.n_test, seed=args.seed)

        save_fewshot_metadata(args.output_dir, task_name,
                              fewshot_samples, fewshot_indices, args.seed)

        fewshot_prefix = build_fewshot_prefix(fewshot_samples)
        sys_prompt     = TASK_SYSTEM_PROMPTS.get(task_name)
        stop_seq       = ["\n"] if task_name in REDSAGE_TASKS else []

        task_summary = {}
        for mode in args.prompt_modes:
            out_file = os.path.join(args.output_dir, f"{task_name}_{mode}_responses.jsonl")
            if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                n = sum(1 for _ in open(out_file))
                print(f"  [RESUME] {mode}: {n} rows already in {out_file}")
                task_summary[mode] = {"samples": n, "resumed": True}
                continue

            if mode == "cot":
                max_tok = COT_TASK_MAX_NEW_TOKENS.get(task_name, COT_MAX_NEW_TOKENS)
            else:
                max_tok = TASK_MAX_NEW_TOKENS.get(task_name, 1024)
            eff_stop = [] if mode == "cot" else stop_seq

            try:
                if task_name == "seceval":
                    n = collect_seceval_with_mode(
                        test_samples, mode, out_file, fewshot_samples,
                        max_tok, model, tokenizer, **api_kwargs)
                else:
                    n = collect_with_mode(
                        task_name, test_samples, mode, out_file,
                        fewshot_prefix, sys_prompt, max_tok, eff_stop,
                        model, tokenizer, **api_kwargs)
                task_summary[mode] = {"samples": n, "output_file": out_file}
            except Exception as e:
                print(f"  ERROR {mode}: {e}")
                task_summary[mode] = {"error": str(e)}

        summary[task_name] = task_summary

    # Save summary
    summary_path = os.path.join(args.output_dir, "sensitivity_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("SENSITIVITY STUDY COMPLETE")
    print(f"Output directory: {args.output_dir}")
    print(f"Summary: {summary_path}")
    for task, info in summary.items():
        if isinstance(info, dict) and "error" in info:
            print(f"  {task}: ERROR — {info['error']}")
        else:
            for mode, minfo in (info or {}).items():
                if "error" in minfo:
                    print(f"  {task}/{mode}: ERROR — {minfo['error']}")
                else:
                    flag = " (resumed)" if minfo.get("resumed") else ""
                    print(f"  {task}/{mode}: {minfo.get('samples')} samples{flag}")


if __name__ == "__main__":
    main()
