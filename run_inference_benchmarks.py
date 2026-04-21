#!/usr/bin/env python3
"""
Collect raw LLM responses from cybersecurity benchmarks without evaluation.
Responses are saved in JSONL format for later evaluation (regex or LLM judge).

Supported benchmarks:
- CTI-Bench (RISys-Lab): MCQ, RCM, VSP, ATE
- MMLU Computer Security
- SECURE: MAET, CWET, KCV
- SecBench
- RedSageMCQ: 5 subsets (Frameworks, Generals, Skills, CLI, Kali)
- CyberMetric-500
- AthenaBench (GitHub JSONL): CKT, RMS, TAA
- SecEval
- CISSP
"""

import os
import json
import torch
import requests
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed


# Optional peft import for LoRA adapter support
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from datetime import datetime

# Optional vLLM import for fast batched inference
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


def get_task_type(task_name: str) -> str:
    """Map task name to evaluation type for LLM judge.

    This centralizes the task type mapping so the judge can read it from
    response metadata without hard-coding every benchmark name.
    """
    task_type_map = {
        # MCQ-style tasks (single choice A/B/C/D)
        "mcq": "mcq",
        "cybermetric": "mcq",
        "cissp": "mcq",
        "mmlu-cs": "mcq",
        "secbench": "mcq",
        "ckt": "ckt",  # 5-option MCQ (A/B/C/D/E)
        "secure_maet": "secure",
        "secure_cwet": "secure",
        "secure_kcv": "secure",
        "redsage_frameworks": "mcq",
        "redsage_generals": "mcq",
        "redsage_skills": "mcq",
        "redsage_cli": "mcq",
        "redsage_kali": "mcq",

        # Multi-select MCQ
        "seceval": "seceval",

        # Structured extraction tasks
        "rcm": "rcm",  # CWE ID extraction (CTI-Bench)
        "athena_rcm": "rcm",  # CWE ID extraction (AthenaBench)
        "vsp": "vsp",  # CVSS vector extraction (CTI-Bench)
        "athena_vsp": "vsp",  # CVSS vector extraction (AthenaBench)
        "ate": "ate",  # MITRE ATT&CK techniques (CTI-Bench)
        "athena_ate": "ate",  # MITRE ATT&CK techniques (AthenaBench, expanded)
        "rms": "rms",  # Risk mitigation strategies
        "taa": "taa",  # Threat actor attribution (AthenaBench)
        "cti_taa": "taa",  # Threat actor attribution (CTI-Bench)
    }
    return task_type_map.get(task_name, "mcq")  # Default to MCQ


def load_model_and_tokenizer(model_path: str, base_model: str = None, is_base: bool = False):
    """Load model (base or fine-tuned) with optional LoRA adapters"""
    if is_base:
        print(f"Loading BASE model from: {model_path}")
        model_path_to_load = model_path
    else:
        print(f"Loading FINE-TUNED model from: {model_path}")
        model_path_to_load = model_path
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model if (base_model and not is_base) else model_path_to_load, 
        trust_remote_code=True
    )
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if is_base:
        model = AutoModelForCausalLM.from_pretrained(
            model_path_to_load,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    elif os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("Loading LoRA adapters...")
        if not HAS_PEFT:
            raise ImportError(
                "PEFT library not found but LoRA adapter detected. "
                "Install peft with: pip install peft"
            )
        if not base_model:
            raise ValueError("base_model must be specified when loading LoRA adapters")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path_to_load,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    return model, tokenizer


def chat_completion_api(endpoint: str, model_name: str, prompt: str = None,
                       api_key: str = "", max_tokens: int = 1024,
                       temperature: float = 0.0, retries: int = 3,
                       system_prompt: str = None, messages: list = None,
                       api_version: str = "2024-12-01-preview") -> str:
    """Call Azure OpenAI chat completions API using the official SDK."""
    from openai import AzureOpenAI

    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"API call failed (attempt {attempt+1}/{retries}): {e}")
                import time; time.sleep(2 ** attempt)
            else:
                print(f"API call failed after {retries} attempts: {e}")
                return f"ERROR: {str(e)}"
    return f"ERROR: exhausted retries"


# Kept for backward compatibility — delegates to chat_completion_api
def chat_responses_api(endpoint: str, model_name: str, prompt: str = None,
                       api_key: str = "", max_tokens: int = 1024,
                       temperature: float = 0.0, retries: int = 3,
                       system_prompt: str = None, messages: list = None) -> str:
    return chat_completion_api(endpoint, model_name, prompt=prompt,
                               api_key=api_key, max_tokens=max_tokens,
                               temperature=temperature, retries=retries,
                               system_prompt=system_prompt, messages=messages)


def generate_response(model, tokenizer, prompt: str = None, max_new_tokens: int = 1024,
                     use_api: bool = False, use_vllm: bool = False, vllm_model=None,
                     api_endpoint: str = None, api_model: str = None, api_key: str = "",
                     batch_size: int = None, system_prompt: str = None,
                     messages: list = None, api_style: str = "chat_completions",
                     **kwargs) -> str:
    """Generate response using local inference, vLLM, or API

    Args:
        batch_size: Ignored (used only for vLLM config, not here)
        system_prompt: Optional system instruction for chat-style models
        **kwargs: Ignored additional parameters for compatibility
    """
    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})

    if use_api:
        return chat_completion_api(
            api_endpoint,
            api_model,
            prompt=prompt,
            api_key=api_key,
            max_tokens=max_new_tokens,
            temperature=0.0,
            system_prompt=system_prompt,
            messages=messages,
        )

    if use_vllm and vllm_model:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        responses = generate_responses_vllm(
            vllm_model,
            [formatted_prompt],
            max_new_tokens,
            temperature=0.0
        )
        return responses[0]

    # Local inference with HuggingFace transformers
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def initialize_vllm(model_path: str, base_model: str = None,
                   gpu_memory_utilization: float = 0.9,
                   max_model_len: int = None,
                   num_gpu_blocks_override: int = None,
                   enforce_eager: bool = False) -> "LLM":
    """Initialize vLLM LLM for fast batch inference

    Args:
        model_path: Path to model directory
        base_model: Base model name (if needed)
        gpu_memory_utilization: GPU memory fraction to use
        max_model_len: Maximum model context length (None = model default)
        num_gpu_blocks_override: Force a specific number of KV cache blocks.
            Required for models whose KV cache profiler returns 0 (e.g. Gemma-4
            with heterogeneous head_dim across sliding/global attention layers).
            Compute as: floor(available_kv_mem_GiB / per_block_mem_GiB).
        enforce_eager: Disable CUDA graph capture and run in eager mode.
            Bypasses graph compilation issues on architectures with mixed
            attention types. Slower but more robust.

    Returns:
        vLLM LLM object
    """
    if not HAS_VLLM:
        raise RuntimeError(
            "vLLM not installed. Install with: pip install vllm\n"
            "Or use --use_api for API-based inference."
        )

    print(f"Initializing vLLM with model: {model_path}")
    print(f"GPU memory utilization: {gpu_memory_utilization*100:.0f}%")
    if max_model_len:
        print(f"Max model len: {max_model_len}")
    if num_gpu_blocks_override:
        print(f"num_gpu_blocks_override: {num_gpu_blocks_override}")
    if enforce_eager:
        print("enforce_eager: True (CUDA graphs disabled)")

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("vLLM requires at least one CUDA GPU but none were detected.")

    vllm_kwargs = dict(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        enforce_eager=enforce_eager,
    )
    if max_model_len:
        vllm_kwargs["max_model_len"] = max_model_len
    if num_gpu_blocks_override:
        # Needed for models where vLLM's profiler computes 0 KV blocks due to
        # heterogeneous attention head dims (e.g. Gemma-4 sliding vs global layers).
        vllm_kwargs["num_gpu_blocks_override"] = num_gpu_blocks_override
    llm = LLM(**vllm_kwargs)

    return llm


def generate_responses_vllm(vllm_llm: "LLM", prompts: list, 
                           max_tokens: int = 1024,
                           temperature: float = 0.0) -> list:
    """Generate responses using vLLM batch inference
    
    Args:
        vllm_llm: vLLM LLM object
        prompts: List of prompts to generate responses for
        max_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature
        
    Returns:
        List of generated responses
    """
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
    )
    
    # Generate responses in batch
    outputs = vllm_llm.generate(prompts, sampling_params)
    
    # Extract text from outputs
    responses = [output.outputs[0].text.strip() for output in outputs]
    return responses


def batch_generate(items, tokenizer, max_new_tokens,
                   use_api=False, use_vllm=False, vllm_model=None,
                   model=None, n_api_workers=1, **api_kwargs) -> list:
    """Generate responses for a list of prompts or message-lists.

    Each element of *items* is either:
      - a plain string prompt, or
      - a list of dicts  (OpenAI-style messages).

    When vLLM is active every prompt is formatted with the tokenizer's chat
    template and submitted as a single batch — this is critical for throughput.
    API mode supports parallel calls via n_api_workers (ThreadPoolExecutor).
    Local-HF mode falls back to sequential per-item calls.
    """
    if use_vllm and vllm_model:
        formatted = []
        for item in items:
            msgs = item if isinstance(item, list) else [{"role": "user", "content": item}]
            try:
                fp = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                fp = item if isinstance(item, str) else msgs[-1].get("content", "")
            formatted.append(fp)
        return generate_responses_vllm(vllm_model, formatted, max_new_tokens)

    def _call_one(item):
        if isinstance(item, list):
            return generate_response(
                model, tokenizer, messages=item,
                max_new_tokens=max_new_tokens,
                use_api=use_api, use_vllm=False, vllm_model=None,
                **api_kwargs,
            )
        else:
            return generate_response(
                model, tokenizer, prompt=item,
                max_new_tokens=max_new_tokens,
                use_api=use_api, use_vllm=False, vllm_model=None,
                **api_kwargs,
            )

    if use_api and n_api_workers > 1:
        responses = [None] * len(items)
        with ThreadPoolExecutor(max_workers=n_api_workers) as executor:
            futures = {executor.submit(_call_one, item): i for i, item in enumerate(items)}
            for future in tqdm(as_completed(futures), total=len(items), desc="API calls"):
                i = futures[future]
                responses[i] = future.result()
        return responses

    # Sequential fallback for local HF inference (or API with n_api_workers=1)
    responses = []
    for item in tqdm(items, desc="Generating") if use_api else items:
        responses.append(_call_one(item))
    return responses


def load_jsonl_dataset(source: str) -> Dataset:
    """Load JSONL dataset from GitHub URL or local file path
    
    Args:
        source: Either a GitHub raw URL or local file path to .jsonl file
    
    Returns:
        HuggingFace Dataset object
    """
    data = []
    
    # Check if source is a URL or local path
    if source.startswith('http://') or source.startswith('https://'):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
    else:
        with open(source, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    
    # Parse JSONL - convert all to strings to avoid PyArrow type conflicts
    for line in lines:
        line = line.strip()
        if line:
            obj = json.loads(line)
            # Convert all values to strings to avoid type conflicts in PyArrow
            cleaned_obj = {}
            for key, value in obj.items():
                if value is None:
                    cleaned_obj[key] = ""
                else:
                    cleaned_obj[key] = str(value)
            data.append(cleaned_obj)
    
    return Dataset.from_list(data)


def collect_huggingface_benchmark(task_name: str, dataset_name: str, subset_name: str,
                                  model, tokenizer, output_file: str, max_samples: int = None,
                                  max_new_tokens: int = 1024, **api_kwargs):
    """Collect responses from HuggingFace datasets (CTI-Bench, SECURE, SecBench, RedSageMCQ)"""
    print(f"\n{'='*70}")
    print(f"Collecting {task_name.upper()} responses")
    print(f"Dataset: {dataset_name}/{subset_name}")
    print(f"{'='*70}")

    # Load dataset
    dataset = load_dataset(dataset_name, subset_name, split="test")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Total samples: {len(dataset)}")

    # SECURE tasks use a different instruction wording (no context in the parenthetical)
    # SecBench uses RedSage's official wording (no "(A, B, C, D)" parenthetical)
    # All other MCQ tasks use the standard instruction
    if task_name == "secbench":
        default_instruction = "You are given multiple choice questions. Answer with the option letter from the given choices directly."
    else:
        default_instruction = "You are given multiple choice questions. Answer with the option letter (A, B, C, D) from the given choices directly."

    # ── Pass 1: build all prompts and metadata ────────────────────────────────
    prompts_list = []
    meta_list = []   # (ground_truth, metadata, original_prompt_str) per sample

    for idx, sample in enumerate(dataset):
        # Normalize field names: Prompt (CTI-Bench/SECURE) > prompt > question
        prompt = sample.get('Prompt') or sample.get('prompt') or sample.get('question')
        if not prompt:
            continue

        # Normalize answer field: GT > solution > answer > label
        ground_truth = sample.get('GT') or sample.get('solution')
        if not ground_truth:
            answer_val = sample.get('answer')
            if answer_val is None:
                answer_val = sample.get('label')
            if answer_val is not None:
                if isinstance(answer_val, int):
                    ground_truth = ['A', 'B', 'C', 'D'][answer_val]
                else:
                    ground_truth = str(answer_val).strip()
            else:
                ground_truth = ""
        else:
            ground_truth = str(ground_truth).strip()

        # SECURE: use Prompt field as-is (already fully formatted)
        if not task_name.startswith("secure_"):
            choices = sample.get('answers') or sample.get('choices') or sample.get('options')
            if choices:
                formatted_prompt = default_instruction + "\n\nQuestion: " + prompt + "\n"
                if isinstance(choices, dict):
                    for key in ['A', 'B', 'C', 'D']:
                        if key in choices:
                            formatted_prompt += f"{key}. {choices[key]}\n"
                elif isinstance(choices, list):
                    choice_letters = ['A', 'B', 'C', 'D']
                    for ci, option in enumerate(choices[:4]):
                        formatted_prompt += f"{choice_letters[ci]}. {option}\n"
                formatted_prompt += "Answer:"
                prompt = formatted_prompt

        metadata = {
            "dataset": dataset_name,
            "subset": subset_name,
            "sample_id": idx,
            "task_type": get_task_type(task_name),
        }
        if 'answers' in sample:
            metadata['choices'] = sample['answers']
        elif 'choices' in sample:
            metadata['choices'] = sample['choices']
        elif 'options' in sample:
            metadata['choices'] = sample['options']

        prompts_list.append(prompt)
        meta_list.append((ground_truth, metadata))

    print(f"Prompts built: {len(prompts_list)}")

    # ── Pass 2: batch generate ────────────────────────────────────────────────
    responses = batch_generate(
        prompts_list, tokenizer, max_new_tokens,
        model=model, **api_kwargs
    )

    # ── Pass 3: assemble and save results ─────────────────────────────────────
    results = []
    for idx, (prompt, (ground_truth, metadata), response) in enumerate(
        zip(prompts_list, meta_list, responses)
    ):
        results.append({
            "task": task_name,
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": metadata,
        })

    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Saved {len(results)} responses to: {output_file}")
    return len(results)


def collect_athenabench_jsonl(task_name: str, jsonl_url: str,
                             model, tokenizer, output_file: str, max_samples: int = None,
                             max_new_tokens: int = 1024, **api_kwargs):
    """Collect responses from AthenaBench GitHub JSONL tasks (CKT, RMS, TAA, ATE, RCM, VSP)"""
    print(f"\n{'='*70}")
    print(f"Collecting {task_name.upper()} responses")
    print(f"JSONL source: {jsonl_url}")
    print(f"{'='*70}")
    
    # Load JSONL dataset
    dataset = load_jsonl_dataset(jsonl_url)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Total samples: {len(dataset)}")

    # ── Pass 1: build prompts and metadata ────────────────────────────────────
    prompts_list = []
    meta_list = []

    for idx, sample in enumerate(dataset):
        question = sample.get('question', '')
        prompt = sample.get('prompt', question)
        ground_truth = sample.get('answer', sample.get('correct_answer', '')).strip()

        metadata = {
            "source": jsonl_url,
            "sample_id": idx,
            "task_type": get_task_type(task_name),
        }
        if 'option_a' in sample:
            metadata['choices'] = {
                'A': sample.get('option_a', ''),
                'B': sample.get('option_b', ''),
                'C': sample.get('option_c', ''),
                'D': sample.get('option_d', ''),
                'E': sample.get('option_e', ''),
            }

        prompts_list.append(prompt)
        meta_list.append((ground_truth, metadata))

    print(f"Prompts built: {len(prompts_list)}")

    # ── Pass 2: batch generate ────────────────────────────────────────────────
    responses = batch_generate(
        prompts_list, tokenizer, max_new_tokens,
        model=model, **api_kwargs
    )

    # ── Pass 3: assemble and save results ─────────────────────────────────────
    results = []
    for idx, (prompt, (ground_truth, metadata), response) in enumerate(
        zip(prompts_list, meta_list, responses)
    ):
        results.append({
            "task": task_name,
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": metadata,
        })

    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Saved {len(results)} responses to: {output_file}")
    return len(results)


def collect_mmlu_cs(model, tokenizer, output_file: str, max_samples: int = None,
                   max_new_tokens: int = 1024, **api_kwargs):
    """Collect responses from MMLU Computer Security using the official 5-shot format.

    Official format from Hendrycks et al. (hendrycks/test):
    - Header: "The following are multiple choice questions (with answers) about computer security."
    - 5 in-context examples from the dev split (each with "Answer: X")
    - Test question appended without the answer
    """
    print(f"\n{'='*70}")
    print("Collecting MMLU-CS responses (official 5-shot format)")
    print(f"{'='*70}")

    # Load dev split for few-shot examples and test split for evaluation
    dev_dataset = load_dataset("lighteval/mmlu", "computer_security", split="dev")
    test_dataset = load_dataset("lighteval/mmlu", "computer_security", split="test")

    if max_samples:
        test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))

    print(f"Dev examples (few-shot): {len(dev_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    choice_letters = ['A', 'B', 'C', 'D']
    subject = "computer security"
    header = f"The following are multiple choice questions (with answers) about {subject}.\n\n"

    def format_example(sample, include_answer: bool) -> str:
        """Format a single MCQ example in MMLU style."""
        choices = sample.get('choices', [])
        answer_idx = sample.get('answer')
        question = sample.get('question', '')

        text = question.strip() + "\n"
        for i, opt in enumerate(choices[:4]):
            text += f"{choice_letters[i]}. {opt}\n"
        text += "Answer:"
        if include_answer and answer_idx is not None:
            label = choice_letters[answer_idx] if isinstance(answer_idx, int) else str(answer_idx)
            text += f" {label}"
        return text

    # Build 5-shot prefix (all dev examples, typically 5 for MMLU)
    few_shot_parts = [header]
    for dev_sample in dev_dataset:
        few_shot_parts.append(format_example(dev_sample, include_answer=True))
        few_shot_parts.append("\n\n")
    few_shot_prefix = "".join(few_shot_parts)

    # ── Pass 1: build prompts ─────────────────────────────────────────────────
    prompts_list = []
    meta_list = []

    for idx, sample in enumerate(test_dataset):
        choices = sample.get('choices', [])
        answer_idx = sample.get('answer')
        if not choices or answer_idx is None:
            continue

        ground_truth = choice_letters[answer_idx] if isinstance(answer_idx, int) else str(answer_idx).strip()
        test_part = format_example(sample, include_answer=False)
        prompt = few_shot_prefix + test_part

        prompts_list.append(prompt)
        meta_list.append((ground_truth, choices))

    print(f"Prompts built: {len(prompts_list)}")

    # ── Pass 2: batch generate ────────────────────────────────────────────────
    responses = batch_generate(
        prompts_list, tokenizer, max_new_tokens,
        model=model, **api_kwargs
    )

    # ── Pass 3: assemble and save results ─────────────────────────────────────
    results = []
    for idx, (prompt, (ground_truth, choices), response) in enumerate(
        zip(prompts_list, meta_list, responses)
    ):
        results.append({
            "task": "mmlu-cs",
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": {
                "dataset": "lighteval/mmlu",
                "subset": "computer_security",
                "sample_id": idx,
                "task_type": "mcq",
                "choices": choices,
                "prompt_style": "5shot_official",
            },
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(results)} responses to: {output_file}")
    return len(results)


def collect_seceval(model, tokenizer, output_file: str, max_samples: int = None,
                   max_new_tokens: int = 1024, **api_kwargs):
    """Collect responses from SecEval using chat-few-shot prompting"""
    print(f"\n{'='*70}")
    print(f"Collecting SecEval responses")
    print(f"{'='*70}")

    dataset_url = "https://huggingface.co/datasets/XuanwuAI/SecEval/resolve/main/questions.json"
    response = requests.get(dataset_url, timeout=30)
    response.raise_for_status()
    questions = response.json()

    if max_samples:
        questions = questions[:max_samples]

    print(f"Total samples: {len(questions)}")

    instruction = "Below are multiple-choice questions concerning cybersecurity. Please select the correct answers and respond with the letters ABCD only."

    chat_few_shot = [
        {
            "role": "user",
            "content": "Question: Which mitigation prevent stack overflow bug? A: Stack Canary. B: ALSR. C: CFI. D: Code Signing."
        },
        {
            "role": "assistant",
            "content": "Answer: ABC",
        },
    ]

    # ── Pass 1: build message-lists and metadata ──────────────────────────────
    messages_list = []
    meta_list = []

    for idx, q in enumerate(questions):
        question = q.get("question", "")
        choices = q.get("choices", [])
        if not question or not choices:
            continue

        question_text = "Question: " + question + " " + " ".join(choices)
        question_text = question_text.replace("\n", " ")

        messages = (
            [{"role": "system", "content": instruction}]
            + chat_few_shot
            + [{"role": "user", "content": question_text}]
        )
        messages_list.append(messages)
        meta_list.append({
            "question": question,
            "choices": choices,
            "id": q.get("id", ""),
            "task_type": "seceval",
            "source": q.get("source", ""),
            "topics": q.get("topics", []),
            "keyword": q.get("keyword", ""),
            "prompt_style": "chat_fewshot",
            "ground_truth": q.get("answer", ""),
        })

    print(f"Prompts built: {len(messages_list)}")

    # ── Pass 2: batch generate ────────────────────────────────────────────────
    responses = batch_generate(
        messages_list, tokenizer, max_new_tokens,
        model=model, **api_kwargs
    )

    # ── Pass 3: assemble and save results ─────────────────────────────────────
    results = []
    for idx, (messages, meta, response) in enumerate(zip(messages_list, meta_list, responses)):
        results.append({
            "task": "seceval",
            "index": idx,
            "prompt": messages,
            "ground_truth": meta.pop("ground_truth"),
            "model_response": response,
            "metadata": meta,
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(results)} responses to: {output_file}")
    return len(results)


def collect_cybermetric(model, tokenizer, output_file: str, max_samples: int = None,
                        max_new_tokens: int = 1024, **api_kwargs):
    """Collect responses from CyberMetric"""
    print(f"\n{'='*70}")
    print("Collecting CyberMetric responses")
    print(f"{'='*70}")

    dataset = load_dataset(
        "RISys-Lab/Benchmarks_CyberSec_CyberMetrics",
        "cyberMetric_500",
        split="test"
    )

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Total samples: {len(dataset)}")

    system_prompt = "You are a security expert who answers questions."

    # ── Pass 1: build prompts ─────────────────────────────────────────────────
    prompts_list = []
    meta_list = []

    for idx, sample in enumerate(dataset):
        question = sample.get("question", "")
        answers = sample.get("answers", {}) or {}
        ground_truth = str(sample.get("solution", "")).strip()

        if not question:
            continue

        if not answers:
            answers = sample.get("choices", {}) or sample.get("options", {}) or {}

        if isinstance(answers, list):
            letters = ["A", "B", "C", "D"]
            answers = {letters[i]: str(opt) for i, opt in enumerate(answers[:4])}

        if not isinstance(answers, dict) or len(answers) == 0:
            continue

        options = ", ".join([f"{key}) {value}" for key, value in answers.items()])
        # Embed system prompt in user message for cybermetric (system_prompt handled in chat template)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Question: {question}\n"
                f"Options: {options}\n\n"
                f"Choose the correct answer (A, B, C, or D) only. "
                f"Always return in this format: 'ANSWER: X'"
            )},
        ]
        prompts_list.append(messages)
        meta_list.append((ground_truth, question, answers, idx))

    print(f"Prompts built: {len(prompts_list)}")

    # ── Pass 2: batch generate ────────────────────────────────────────────────
    responses = batch_generate(
        prompts_list, tokenizer, max_new_tokens,
        model=model, **api_kwargs
    )

    # ── Pass 3: assemble and save results ─────────────────────────────────────
    results = []
    for messages, (ground_truth, question, answers, sample_idx), response in zip(
        prompts_list, meta_list, responses
    ):
        results.append({
            "task": "cybermetric",
            "index": sample_idx,
            "prompt": messages[-1]["content"],
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": {
                "dataset": "RISys-Lab/Benchmarks_CyberSec_CyberMetrics",
                "subset": "cyberMetric_500",
                "sample_id": sample_idx,
                "task_type": "mcq",
                "question": question,
                "choices": answers,
            },
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(results)} responses to: {output_file}")
    return len(results)
    
def collect_cissp(model, tokenizer, output_file: str, dataset_path: str = None,
                 max_samples: int = None, max_new_tokens: int = 1024, **api_kwargs):
    """Collect responses from CISSP"""
    print(f"\n{'='*70}")
    print(f"Collecting CISSP responses")
    print(f"{'='*70}")
    
    # Load CISSP dataset
    if dataset_path is None:
        raise ValueError("CISSP dataset path must be provided via --cissp_path argument")
    
    if dataset_path.startswith('http'):
        response = requests.get(dataset_path)
        data = response.json()
    else:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    
    # Handle different data structures (list or dict with 'questions'/'items' key)
    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict):
        questions = data.get('questions') or data.get('items') or data.get('data') or []
    else:
        questions = []
    
    if max_samples:
        questions = questions[:max_samples]
    
    print(f"Total samples: {len(questions)}")
    
    # Collect responses
    results = []
    total = len(questions)
    for idx, q in enumerate(tqdm(questions, desc="Collecting CISSP")):
        # Extract question and choices flexibly
        question = q.get('question') or q.get('Prompt') or ""
        if not question:
            continue
        
        # Handle different choice formats
        choices = {}
        if isinstance(q.get('answers'), dict):
            choices = q['answers']
        elif isinstance(q.get('options'), list):
            # Convert list to dict
            for i, choice in enumerate(q['options']):
                if i < 4:  # A, B, C, D
                    choices[chr(65+i)] = choice
        elif isinstance(q.get('options'), dict):
            choices = q['options']
        elif isinstance(q.get('choices'), list):
            # Convert list to dict
            for i, choice in enumerate(q['choices']):
                if i < 4:
                    choices[chr(65+i)] = choice
        else:
            # Try direct A/B/C/D keys
            for label in ['A', 'B', 'C', 'D']:
                if label in q:
                    choices[label] = q[label]
        
        # Get correct answer flexibly
        correct_answer = ""
        for key in ['solution', 'answer', 'GT', 'correct_answer']:
            if key in q:
                correct_answer = str(q[key]).strip()
                break
        
        # Skip if missing data
        if not choices or not correct_answer:
            continue
        
        # Format prompt
        prompt = f"{question}\n\n"
        for label in sorted(choices.keys()):
            prompt += f"{label}. {choices[label]}\n"
        prompt += "\nAnswer with the letter only:"
        
        # Generate response
        response = generate_response(model, tokenizer, prompt, max_new_tokens=max_new_tokens, **api_kwargs)

        # Store raw result
        result = {
            "task": "cissp",
            "index": idx,
            "prompt": prompt,
            "ground_truth": correct_answer,
            "model_response": response,
            "metadata": {
                "question": question,
                "choices": choices,
                "domain": q.get('domain', ''),
                "task_type": "mcq"  # Single-choice MCQ
            }
        }
        results.append(result)
    
    # Save to JSONL
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"✓ Saved {len(results)} responses to: {output_file}")
    return len(results)


def main():
    parser = argparse.ArgumentParser(description="Collect raw benchmark responses without evaluation")
    
    # Model loading options
    parser.add_argument("--model_path", type=str, help="Path to model (base or fine-tuned) - for local inference")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name (required for LoRA)")
    parser.add_argument("--is_base", action="store_true", help="Evaluate base model (pre-training)")
    
    # API endpoint options
    parser.add_argument("--use_api", action="store_true", help="Use API endpoint instead of local model")
    parser.add_argument("--api_endpoint", type=str, help="OpenAI-compatible API endpoint")
    parser.add_argument("--api_model", type=str, help="Model name for API endpoint")
    parser.add_argument("--api_key", type=str, default="", help="API key if needed")
    parser.add_argument("--api_style", type=str, default="chat_completions",
                       choices=["chat_completions", "azure_responses"],
                       help="API request format: 'chat_completions' (default, OpenAI-compatible) "
                            "or 'azure_responses' (Azure OpenAI /openai/responses endpoint)")
    parser.add_argument("--n_api_workers", type=int, default=8,
                       help="Number of parallel API workers for --use_api mode (default: 8)")
    parser.add_argument("--skip_completed", action="store_true",
                       help="Skip tasks where output JSONL already exists")
    
    # vLLM options for fast batch inference
    parser.add_argument("--use_vllm", action="store_true", 
                       help="Use vLLM for fast batch inference (requires vllm package)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory fraction to use with vLLM (0.0-1.0, default: 0.9)")
    parser.add_argument("--max_model_len", type=int, default=None,
                       help="Maximum model context length for vLLM (default: use model's native context)")
    parser.add_argument("--num_gpu_blocks_override", type=int, default=None,
                       help="Override number of GPU KV cache blocks. Use when vLLM's profiler returns 0 "
                            "blocks (heterogeneous attention, e.g. Gemma-4). "
                            "Estimate: floor(available_kv_mem_GiB / per_block_mem_GiB).")
    parser.add_argument("--enforce_eager", action="store_true",
                       help="Disable CUDA graph capture (enforce eager mode). Slower but avoids "
                            "compilation failures on architectures with mixed attention types.")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for vLLM inference (default: 16)")
    
    # Collection options
    parser.add_argument("--tasks", nargs="+", default=["mcq", "rcm", "vsp", "ate"],
                       help="Tasks to collect: mcq, rcm, vsp, ate, cti_taa, "
                            "ckt, rms, taa, athena_ate, athena_rcm, athena_vsp, "
                            "mmlu-cs, secure_maet, secure_cwet, secure_kcv, secbench, "
                            "redsage_frameworks, redsage_generals, redsage_skills, "
                            "redsage_cli, redsage_kali, cybermetric, seceval, cissp. "
                            "Note: cti_taa=CTI-Bench TAA (50 items), taa=AthenaBench TAA (100 items), "
                            "athena_ate/rcm/vsp=AthenaBench expanded extraction tasks.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for JSONL files")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples per task (for testing)")
    parser.add_argument("--max_tokens_config", type=str, default=None,
                       help="Path to calibration JSON file with per-task max_tokens values "
                            "(output of calibrate_max_tokens.py). If not provided, uses 1024 for all tasks.")

    # Dataset paths
    parser.add_argument("--cissp_path", type=str, default=None, help="Path to CISSP dataset JSON file")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_api and args.use_vllm:
        parser.error("Cannot use both --use_api and --use_vllm. Choose one inference method.")
    
    if args.use_api:
        if not args.api_endpoint or not args.api_model:
            parser.error("--api_endpoint and --api_model are required when --use_api is set")
        model = None
        tokenizer = None
        vllm_model = None
    elif args.use_vllm:
        if not args.model_path:
            parser.error("--model_path is required for vLLM inference")
        vllm_model = initialize_vllm(
            args.model_path,
            args.base_model,
            args.vllm_gpu_memory_utilization,
            args.max_model_len,
            num_gpu_blocks_override=args.num_gpu_blocks_override,
            enforce_eager=args.enforce_eager,
        )
        model = None
        # Load tokenizer separately for chat-template formatting (vLLM doesn't expose this)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        if not args.model_path:
            parser.error("--model_path is required for local inference (or use --use_api or --use_vllm)")
        # Load model for local inference
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model, args.is_base)
        vllm_model = None
    
    # Generate output directory name
    if args.output_dir is None:
        if args.use_api:
            model_name = args.api_model.rstrip('/').split('/')[-1]
        else:
            model_name = args.model_path.rstrip('/').split('/')[-1]
        model_name = model_name.replace('/', '-').replace('\\', '-')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"responses_{model_name}_{timestamp}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}")
    
    # Save metadata
    metadata = {
        "model_path": args.model_path if not args.use_api else args.api_model,
        "evaluation_mode": "api" if args.use_api else ("vllm" if args.use_vllm else "local"),
        "is_base_model": args.is_base,
        "timestamp": datetime.now().isoformat(),
        "tasks": args.tasks,
        "max_samples": args.max_samples
    }
    if args.use_vllm:
        metadata["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
        metadata["batch_size"] = args.batch_size
        metadata["max_model_len"] = args.max_model_len
        if args.num_gpu_blocks_override:
            metadata["num_gpu_blocks_override"] = args.num_gpu_blocks_override
        if args.enforce_eager:
            metadata["enforce_eager"] = True
    
    with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Load per-task max_tokens from calibration JSON (if provided)
    task_max_tokens = {}
    if args.max_tokens_config:
        with open(args.max_tokens_config, 'r') as f:
            calib = json.load(f)
        # Top-level keys are task names with recommended token counts
        for key, val in calib.items():
            if key not in ("model", "is_thinking_model", "tasks") and isinstance(val, int):
                task_max_tokens[key] = val
        print(f"Loaded per-task max_tokens from: {args.max_tokens_config}")
        print(f"  Tasks configured: {list(task_max_tokens.keys())}")

    # Prepare inference kwargs (api_kwargs name kept for backward compatibility)
    api_kwargs = {
        'use_api': args.use_api,
        'use_vllm': args.use_vllm,
        'vllm_model': vllm_model,
        'batch_size': args.batch_size,
        'api_endpoint': args.api_endpoint,
        'api_model': args.api_model,
        'api_key': args.api_key,
        'api_style': args.api_style,
        'n_api_workers': args.n_api_workers if args.use_api else 1,
    }

    # Task configurations
    # Format: "task_name": (dataset_type, dataset_name, subset_or_url)
    task_map = {
        # RISys-Lab CTI-Bench (HuggingFace)
        "mcq": ("hf", "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-mcq"),
        "rcm": ("hf", "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-rcm"),
        "vsp": ("hf", "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-vsp"),
        "ate": ("hf", "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-ate"),
        # cti_taa: not available in RISys-Lab HF mirror (only ate/mcq/rcm/vsp); use AthenaBench taa instead
        
        # Other HuggingFace benchmarks
        "mmlu-cs": ("hf", "lighteval/mmlu", "computer_security"),
        "secure_maet": ("hf", "RISys-Lab/Benchmarks_CyberSec_SECURE", "MAET"),
        "secure_cwet": ("hf", "RISys-Lab/Benchmarks_CyberSec_SECURE", "CWET"),
        "secure_kcv": ("hf", "RISys-Lab/Benchmarks_CyberSec_SECURE", "KCV"),
        "secbench": ("hf", "RISys-Lab/Benchmarks_CyberSec_SecBench", "MCQs_English"),
        
        # RedSageMCQ (5 subsets, 30K total samples)
        "redsage_frameworks": ("hf", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_knowledge_frameworks"),
        "redsage_generals": ("hf", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_knowledge_generals"),
        "redsage_skills": ("hf", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_skills"),
        "redsage_cli": ("hf", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_tools_cli"),
        "redsage_kali": ("hf", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_tools_kali"),
        
        # AthenaBench GitHub JSONL tasks (AthenaBench TAA is expanded version: 100 items)
        "ckt": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-ckt-3k.jsonl", None),
        "rms": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-rms.jsonl", None),
        "taa": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-taa.jsonl", None),
        # AthenaBench expanded extraction tasks (not in CTI-Bench)
        "athena_ate": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-ate.jsonl", None),
        "athena_rcm": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-rcm.jsonl", None),
        "athena_vsp": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-vsp.jsonl", None),
    }
    
    # Collect responses for each task
    summary = {}
    
    for task_name in args.tasks:
        output_file = os.path.join(args.output_dir, f"{task_name}_responses.jsonl")
        # Per-task token budget from calibration config (fallback: 1024)
        task_tokens = task_max_tokens.get(task_name, 1024)

        if args.skip_completed and os.path.exists(output_file):
            print(f"\n[SKIP] {task_name} — output already exists: {output_file}")
            continue

        try:
            if task_name == "mmlu-cs":
                # MMLU-CS uses a dedicated 5-shot collector (official Hendrycks format)
                count = collect_mmlu_cs(model, tokenizer, output_file,
                                        args.max_samples, task_tokens, **api_kwargs)
            elif task_name in task_map:
                # Handle mapped tasks based on type
                task_type, source, subset_or_url = task_map[task_name]

                if task_type == "hf":
                    # HuggingFace datasets (CTI-Bench, SECURE, SecBench, RedSageMCQ)
                    count = collect_huggingface_benchmark(task_name, source, subset_or_url,
                                                          model, tokenizer, output_file,
                                                          args.max_samples, task_tokens, **api_kwargs)
                elif task_type == "jsonl":
                    # AthenaBench GitHub JSONL tasks
                    count = collect_athenabench_jsonl(task_name, source,
                                                      model, tokenizer, output_file,
                                                      args.max_samples, task_tokens, **api_kwargs)
            elif task_name == "seceval":
                count = collect_seceval(model, tokenizer, output_file,
                                       args.max_samples, task_tokens, **api_kwargs)
            elif task_name == "cybermetric":
                count = collect_cybermetric(model, tokenizer, output_file,
                                        args.max_samples, task_tokens, **api_kwargs)
            elif task_name == "cissp":
                count = collect_cissp(model, tokenizer, output_file, args.cissp_path,
                                     args.max_samples, task_tokens, **api_kwargs)
            else:
                print(f"Unknown task: {task_name}, skipping...")
                continue
            
            summary[task_name] = {
                "samples_collected": count,
                "output_file": output_file
            }
        
        except Exception as e:
            print(f"Error collecting {task_name}: {e}")
            summary[task_name] = {"error": str(e)}
    
    # Save summary
    summary_file = os.path.join(args.output_dir, "collection_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {args.output_dir}")
    print(f"Summary: {summary_file}")
    print("\nCollected responses:")
    for task, info in summary.items():
        if 'error' in info:
            print(f"  {task}: ERROR - {info['error']}")
        else:
            print(f"  {task}: {info['samples_collected']} samples → {info['output_file']}")


if __name__ == "__main__":
    main()
