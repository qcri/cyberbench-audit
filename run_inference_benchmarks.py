#!/usr/bin/env python3
"""
Collect model outputs from cybersecurity benchmark tasks in a benchmark-faithful format.

This script focuses on response collection, not final benchmark aggregation. For most
tasks, it saves raw model responses to JSONL so that benchmark-specific evaluators
(regex, exact match, LLM judge, or custom scoring scripts) can run later.

Some tasks also include local-HF scoring-style collection modes when the original
benchmark relies on log-likelihood rather than generated text. For example,
mmlu-cs-logprobs computes answer-choice logprobs locally to better mirror the
original MMLU evaluation.

Supported benchmark families:
- CTI-Bench original TSVs: MCQ, RCM, RCM-2021, VSP, ATE, TAA
- AthenaBench original JSONL: CKT, ATE, RCM, RMS, VSP, TAA
- SECURE original TSVs: MAET, CWET, KCV
- SecEval official chat few-shot mode
- CyberMetric-500 original evaluator prompt
- MMLU Computer Security:
  - mmlu-cs: original 5-shot prompt with generated response collection
  - mmlu-cs-logprobs: local-HF official-style logprob scoring
- SecBench MCQ original data with reconstructed prompt
- RedSageMCQ five subsets using RedSage cybersec_prompt_fn-style generation
- CISSP custom/local dataset collection
"""

import os
import json

import pandas as pd
import numpy as np
from io import StringIO
import torch
import requests
import argparse
import re
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


SEVENLLM_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_input_qwen": (
        "<|im_start|>user\nBelow is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}### Response:<|im_end|>\n<|im_start|>assistant\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

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
   
# ---------------------------------------------------------------------
# MMLU helpers
# ---------------------------------------------------------------------
MMLU_CHOICES = ["A", "B", "C", "D"]

def format_mmlu_subject(subject: str) -> str:
    # Faithful to original MMLU format_subject():
    # "computer_security" -> " computer security"
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def mmlu_answer_letter(answer_val):
    if isinstance(answer_val, int):
        return MMLU_CHOICES[answer_val]
    return str(answer_val).strip()


def format_mmlu_example(sample: dict, include_answer: bool = True) -> str:
    prompt = sample["question"]
    choices = sample["choices"]

    for j in range(len(choices)):
        prompt += "\n{}. {}".format(MMLU_CHOICES[j], choices[j])

    prompt += "\nAnswer:"

    if include_answer:
        prompt += " {}\n\n".format(mmlu_answer_letter(sample["answer"]))

    return prompt


def gen_mmlu_prompt(dev_rows: list, subject: str, k: int = -1) -> str:
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_mmlu_subject(subject)
    )

    if k == -1:
        k = len(dev_rows)

    for i in range(k):
        prompt += format_mmlu_example(dev_rows[i], include_answer=True)

    return prompt


def build_mmlu_full_prompt(dev_rows: list, test_sample: dict, subject: str, ntrain: int = 5) -> tuple:
    k = min(ntrain, len(dev_rows))
    prompt_end = format_mmlu_example(test_sample, include_answer=False)
    train_prompt = gen_mmlu_prompt(dev_rows, subject, k)
    prompt = train_prompt + prompt_end
    return prompt, k


def score_mmlu_next_token_local(model, tokenizer, prompt: str):
    """
    Official-style MMLU scoring for local HF models.

    Original OpenAI path:
    - max_tokens=1
    - logprobs=100
    - temperature=0
    - echo=True
    - compare logprobs of " A", " B", " C", " D"
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]
    next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)

    lprobs = []
    token_info = {}

    for ans in MMLU_CHOICES:
        token_text = " " + ans
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)

        if len(token_ids) == 1:
            lp = next_token_logprobs[token_ids[0]].item()
            token_info[ans] = {
                "token_text": token_text,
                "token_ids": token_ids,
                "scoring_note": "single-token exact",
            }
        else:
            lp = next_token_logprobs[token_ids[0]].item()
            token_info[ans] = {
                "token_text": token_text,
                "token_ids": token_ids,
                "scoring_note": "multi-token fallback scored first token only",
            }

        lprobs.append(lp)

    lprobs_np = np.array(lprobs)
    probs = np.exp(lprobs_np - np.max(lprobs_np))
    probs = probs / probs.sum()

    pred = MMLU_CHOICES[int(np.argmax(lprobs_np))]

    return {
        "prediction": pred,
        "logprobs": {MMLU_CHOICES[i]: float(lprobs_np[i]) for i in range(len(MMLU_CHOICES))},
        "probs": {MMLU_CHOICES[i]: float(probs[i]) for i in range(len(MMLU_CHOICES))},
        "token_info": token_info,
    }

# ---------------------------------------------------------------------
# RedSageMCQ helpers
# ---------------------------------------------------------------------
REDSAGE_CHOICES = ["A", "B", "C", "D"]

def build_redsage_prompt(sample: dict, include_context: bool = False) -> str:
    """
    Build the RedSageMCQ prompt using the RedSage cybersec_prompt_fn layout.

    RedSageMCQTask defaults to include_context=False for the released five MCQ
    subsets, so this function does not include the row's content field unless
    explicitly requested.

    This prompt is used by both RedSage's default loglikelihood tasks and its
    _em generative tasks. This script currently uses it for _em-style response
    collection.
    """
    content = sample.get("content", "")
    question = sample["question"]
    answers_dict = sample["answers"]

    query_parts = []

    if include_context and content:
        query_parts.append("Context: " + content)

    query_parts.append("Question: " + question)

    choices_str_parts = []
    for letter in REDSAGE_CHOICES:
        choice_text = answers_dict.get(letter)
        if choice_text is None:
            raise ValueError(f"Missing answer for choice {letter} in sample: {sample}")
        choices_str_parts.append(f"{letter}. {choice_text}")

    instructions = "You are given multiple choice questions. Answer with the option letter (A, B, C, D) from the given choices directly."

    return (
        instructions
        + "\n"
        + "\n\n".join(query_parts)
        + "\n"
        + "\n".join(choices_str_parts)
        + "\nAnswer:"
    )


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

    if earliest is None:
        return text

    return text[:earliest].strip()

# Helpers:
def load_json_dataset(source: str):
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        return response.json()

    with open(source, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_concatenated_json_objects(source: str) -> list:
    """Load SecBench-style data where JSON objects may be newline-separated or concatenated."""
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        text = response.text
    else:
        with open(source, "r", encoding="utf-8") as f:
            text = f.read()

    decoder = json.JSONDecoder()
    idx = 0
    data = []

    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break

        obj, end = decoder.raw_decode(text, idx)
        data.append(obj)
        idx = end

    return data    

#To load TSV datasets (for CTI-Bench)
def load_tsv_dataset(source: str) -> Dataset:
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), sep="\t", encoding="utf-8")
    else:
        df = pd.read_csv(source, sep="\t", encoding="utf-8")

    # Keep values as strings where possible, but preserve missing fields.
    data = df.replace({np.nan: None}).to_dict(orient="records")
    return Dataset.from_list(data)


def get_task_type(task_name: str) -> str:
    task_type_map = {
        # CTI-Bench
        "ctibench_mcq": "mcq",
        "ctibench_rcm": "rcm",
        "ctibench_rcm_2021": "rcm",
        "ctibench_vsp": "vsp",
        "ctibench_ate": "ate",
        "ctibench_taa": "taa",

        # Backward-compatible CTI aliases
        "mcq": "mcq",
        "rcm": "rcm",
        "vsp": "vsp",
        "ate": "ate",
        "cti_taa": "taa",

        # AthenaBench
        "athenabench_ckt": "ckt",
        "athenabench_ate": "ate",
        "athenabench_rcm": "rcm",
        "athenabench_rms": "rms",
        "athenabench_vsp": "vsp",
        "athenabench_taa": "taa",

        # Backward-compatible Athena aliases
        "ckt": "ckt",
        "rms": "rms",
        "taa": "taa",

        # SECURE
        "secure_maet": "secure_mcq",
        "secure_cwet": "secure_mcq",
        "secure_kcv": "secure_mcq",
        "secure_cpst": "secure_saq",
        "secure_rert": "secure_saq",
        "secure_vood": "secure_tf",

        # Other MCQ-style
        "cybermetric": "mcq",
        "cissp": "mcq",
        "mmlu_cs": "mcq",
        "mmlu-cs": "mcq",
        "mmlu-cs-logprobs": "mcq",
        "secbench_mcq": "mcq",
        "secbench": "mcq",

        # RedSage
        "redsage_frameworks": "mcq",
        "redsage_generals": "mcq",
        "redsage_skills": "mcq",
        "redsage_cli": "mcq",
        "redsage_kali": "mcq",

        # Multi-select
        "seceval": "seceval",
        
        # Structured extraction tasks
        "rcm": "rcm",  # CWE ID extraction
        "vsp": "vsp",  # CVSS vector extraction
        "ate": "ate",  # MITRE ATT&CK techniques
        "rms": "rms",  # Risk mitigation strategies
        "taa": "taa",  # Threat actor attribution (AthenaBench)
        "cti_taa": "taa",  # Threat actor attribution (CTI-Bench)
        # SEvenLLM-Bench tasks (structured JSON extraction)
        "sevenllm": "sevenllm",  # Multi-category CTI extraction tasks
    }
    return task_type_map.get(task_name, "mcq")


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
                       temperature: float = 0.0, top_p: float = 1.0,
                       seed: int = None, retries: int = 3,
                       system_prompt: str = None, messages: list = None) -> str:
    """Call OpenAI-compatible API endpoint"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if seed is not None:
        payload["seed"] = seed
    
    for attempt in range(retries):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                print(f"API call failed (attempt {attempt+1}/{retries}): {e}")
                continue
            else:
                print(f"API call failed after {retries} attempts: {e}")
                return f"ERROR: {str(e)}"


def generate_response(model, tokenizer, prompt: str = None, max_new_tokens: int = 1024,
                     use_api: bool = False, use_vllm: bool = False, vllm_model=None,
                     api_endpoint: str = None, api_model: str = None, api_key: str = "",
                     batch_size: int = None, system_prompt: str = None,
                     messages: list = None, task_name: str = None, **kwargs) -> str:
    """Generate response using local inference, vLLM, or API

    Args:
        batch_size: Ignored (used only for vLLM config, not here)
        system_prompt: Optional system instruction for chat-style models
        **kwargs: Ignored additional parameters for compatibility
    """

    # SEVENLLM path: use raw prompt directly
    if task_name == "sevenllm":
        if use_api:
            return chat_completion_api(
                api_endpoint,
                api_model,
                prompt=prompt,
                api_key=api_key,
                max_tokens=max_new_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

        if use_vllm and vllm_model:
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            outputs = vllm_model.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
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
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            system_prompt=system_prompt,
            messages=messages,
        )

    if use_vllm and vllm_model:
        if tokenizer is not None:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # vLLM raw-prompt fallback when tokenizer is not loaded separately.
            if prompt is not None:
                formatted_prompt = prompt
            else:
                formatted_prompt = "\n".join(m.get("content", "") for m in messages)

        responses = generate_responses_vllm(
            vllm_model,
            [formatted_prompt],
            max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
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

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def initialize_vllm(model_path: str, base_model: str = None, 
                   gpu_memory_utilization: float = 0.9) -> "LLM":
    """Initialize vLLM LLM for fast batch inference
    
    Args:
        model_path: Path to model directory
        base_model: Base model name (if needed)
        gpu_memory_utilization: GPU memory fraction to use
        
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
    
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("vLLM requires at least one CUDA GPU but none were detected.")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True
    )
    
    return llm


def generate_responses_vllm(vllm_llm: "LLM", prompts: list,
                           max_tokens: int = 1024,
                           temperature: float = 0.0,
                           top_p: float = 1.0,
                           seed: int = None) -> list:
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
            top_p=top_p,
            seed=seed,
        )
    
    # Generate responses in batch
    outputs = vllm_llm.generate(prompts, sampling_params)
    
    # Extract text from outputs
    responses = [output.outputs[0].text.strip() for output in outputs]
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
                                  **api_kwargs):
    """Generic HuggingFace dataset collector.

    This is a fallback collector for HF-hosted MCQ-style datasets that do not yet have
    a benchmark-specific prompt/evaluation implementation in this script.

    Do not use this for tasks where we have implemented a faithful collector, such as
    MMLU, RedSageMCQ, SECURE, CTI-Bench, or SecBench.
    """
    print(f"\n{'='*70}")
    print(f"Collecting {task_name.upper()} responses")
    print(f"Dataset: {dataset_name}/{subset_name}")
    print(f"{'='*70}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, subset_name, split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Total samples: {len(dataset)}")
    
    # Collect responses
    results = []
    total = len(dataset)
    for idx, sample in enumerate(tqdm(dataset, desc=f"Collecting {task_name.upper()}")):
        # Normalize field names to standard format
        # Priority: Prompt (CTI-Bench) > prompt (SECURE) > question (others)
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
                # Handle MMLU-CS format: answer is integer index
                if isinstance(answer_val, int):
                    ground_truth = ['A', 'B', 'C', 'D'][answer_val]
                else:
                    ground_truth = str(answer_val).strip()
            else:
                ground_truth = ""
        else:
            ground_truth = str(ground_truth).strip()
        
        # Format prompt with choices if available (MCQ-style)
        # Check for choices in this order: answers > choices > options
        choices = sample.get('answers') or sample.get('choices') or sample.get('options')
        if choices:
            # Format as MCQ with 4 options
            instruction = "You are given multiple choice questions. Answer with the option letter (A, B, C, D) from the given choices directly."
            formatted_prompt = instruction + "\n\nQuestion: " + prompt + "\n"
            
            if isinstance(choices, dict):
                # Dict format: {"A": "option1", "B": "option2", ...} (CyberMetric, etc.)
                for key in ['A', 'B', 'C', 'D']:
                    if key in choices:
                        formatted_prompt += f"{key}. {choices[key]}\n"
            elif isinstance(choices, list):
                # List format: ["option1", "option2", "option3", "option4"] (MMLU-CS, etc.)
                choice_letters = ['A', 'B', 'C', 'D']
                for choice_idx, option in enumerate(choices[:4]):  # Take first 4 options
                    formatted_prompt += f"{choice_letters[choice_idx]}. {option}\n"
            
            formatted_prompt += "Answer:"
            prompt = formatted_prompt
        
        # Generate response
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        
        # Prepare metadata with additional info
        metadata = {
            "dataset": dataset_name,
            "subset": subset_name,
            "sample_id": idx,
            "task_type": get_task_type(task_name)  # Add task type for judge
        }
        
        # Add choices if available
        if 'answers' in sample:
            metadata['choices'] = sample['answers']
        elif 'choices' in sample:
            metadata['choices'] = sample['choices']
        elif 'options' in sample:
            metadata['choices'] = sample['options']
        
        # Store raw result
        result = {
            "task": task_name,
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": metadata
        }
        results.append(result)
    
    # Save to JSONL
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"✓ Saved {len(results)} responses to: {output_file}")
    return len(results)

def collect_mmlu_generation(task_name: str, dataset_name: str, subset_name: str,
                            model, tokenizer, output_file: str, max_samples: int = None,
                            **api_kwargs):
    """
    Collect raw MMLU responses using the original MMLU 5-shot prompt format.

    Prompt is faithful. Scoring is not official MMLU because official MMLU
    uses logprobs over answer options, not generated text parsing.
    """
    print(f"\n{'='*70}")
    print(f"Collecting {task_name.upper()} responses with MMLU 5-shot prompt")
    print(f"Dataset: {dataset_name}/{subset_name}")
    print(f"{'='*70}")

    dev_dataset = load_dataset(dataset_name, subset_name, split="dev")
    test_dataset = load_dataset(dataset_name, subset_name, split="test")

    dev_rows = [dict(x) for x in dev_dataset]
    test_rows = [dict(x) for x in test_dataset]

    if max_samples:
        test_rows = test_rows[:max_samples]

    print(f"Dev examples: {len(dev_rows)}")
    print(f"Test samples: {len(test_rows)}")

    results = []

    for idx, sample in enumerate(tqdm(test_rows, desc=f"Collecting {task_name.upper()}")):
        prompt, k = build_mmlu_full_prompt(dev_rows, sample, subset_name, ntrain=5)
        ground_truth = mmlu_answer_letter(sample["answer"])

        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=5,
            temperature=0.0,
            **api_kwargs,
        )

        result = {
            "task": task_name,
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": {
                "dataset": dataset_name,
                "subset": subset_name,
                "sample_id": idx,
                "task_type": "mcq",
                "prompt_mode": "official_mmlu_5shot_prompt",
                "official_inference_script": "hendrycks/test/evaluate.py",
                "ntrain": k,
                "generation_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 5,
                    "official_max_tokens": 1,
                    "official_logprobs": 100,
                    "official_echo": True,
                },
                "official_scoring_note": (
                    "Official MMLU scoring selects the highest-logprob answer among "
                    "' A', ' B', ' C', and ' D'. This mode preserves the prompt but "
                    "collects generated text for later parsing."
                ),
                "question": sample.get("question", ""),
                "choices": sample.get("choices", []),
                "original_fields": sample,
            },
        }
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(results)} MMLU generation responses to: {output_file}")
    return len(results)


def collect_mmlu_logprobs(task_name: str, dataset_name: str, subset_name: str,
                          model, tokenizer, output_file: str, max_samples: int = None,
                          **api_kwargs):
    """
    Collect MMLU predictions using official-style logprob scoring.

    This is closest to the original MMLU implementation, but currently supports
    local HF models only because API/vLLM paths in this script do not expose
    equivalent Completion echo=True logprobs.
    """
    print(f"\n{'='*70}")
    print(f"Collecting {task_name.upper()} with official-style MMLU logprob scoring")
    print(f"Dataset: {dataset_name}/{subset_name}")
    print(f"{'='*70}")

    if api_kwargs.get("use_api") or api_kwargs.get("use_vllm"):
        raise ValueError(
            "mmlu_logprobs currently supports local HF models only. "
            "Official MMLU requires logprobs over ' A', ' B', ' C', and ' D'. "
            "Your current API/vLLM generation paths do not expose the same Completion logprobs behavior."
        )

    dev_dataset = load_dataset(dataset_name, subset_name, split="dev")
    test_dataset = load_dataset(dataset_name, subset_name, split="test")

    dev_rows = [dict(x) for x in dev_dataset]
    test_rows = [dict(x) for x in test_dataset]

    if max_samples:
        test_rows = test_rows[:max_samples]

    print(f"Dev examples: {len(dev_rows)}")
    print(f"Test samples: {len(test_rows)}")

    results = []

    for idx, sample in enumerate(tqdm(test_rows, desc=f"Collecting {task_name.upper()}")):
        prompt, k = build_mmlu_full_prompt(dev_rows, sample, subset_name, ntrain=5)
        ground_truth = mmlu_answer_letter(sample["answer"])

        score_info = score_mmlu_next_token_local(model, tokenizer, prompt)
        prediction = score_info["prediction"]

        result = {
            "task": task_name,
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": prediction,
            "metadata": {
                "dataset": dataset_name,
                "subset": subset_name,
                "sample_id": idx,
                "task_type": "mcq",
                "prompt_mode": "official_mmlu_5shot_prompt",
                "official_inference_script": "hendrycks/test/evaluate.py",
                "ntrain": k,
                "generation_params": {
                    "max_tokens": 1,
                    "logprobs": 100,
                    "temperature": 0,
                    "echo": True,
                    "local_hf_note": (
                        "Implemented by computing next-token logits for answer tokens "
                        "' A', ' B', ' C', and ' D'."
                    ),
                },
                "official_scoring": "argmax logprob among ' A', ' B', ' C', and ' D'",
                "prediction": prediction,
                "correct": prediction == ground_truth,
                "choice_logprobs": score_info["logprobs"],
                "choice_probs": score_info["probs"],
                "choice_token_info": score_info["token_info"],
                "question": sample.get("question", ""),
                "choices": sample.get("choices", []),
                "original_fields": sample,
            },
        }
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(results)} MMLU logprob predictions to: {output_file}")
    return len(results)



def collect_ctibench_tsv(task_name: str, tsv_url: str,
                         model, tokenizer, output_file: str,
                         max_samples: int = None, **api_kwargs):
    print(f"\n{'='*70}")
    print(f"Collecting {task_name.upper()} responses")
    print(f"TSV source: {tsv_url}")
    print(f"{'='*70}")

    dataset = load_tsv_dataset(tsv_url)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Total samples: {len(dataset)}")

    results = []

    for idx, sample in enumerate(tqdm(dataset, desc=f"Collecting {task_name.upper()}")):
        sample = dict(sample)

        # Faithful to CTI-Bench: use Prompt column verbatim.
        prompt = sample.get("Prompt")
        if not prompt:
            continue

        # CTI-Bench mostly uses GT, but TAA may need special eval assets.
        ground_truth = str(sample.get("GT", "") or "").strip()

        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=2048,
            system_prompt="You are a cybersecurity expert specializing in cyberthreat intelligence.",
            temperature=0.0,
            top_p=1.0,
            seed=42,
            **api_kwargs,
        )

        metadata = {
            "dataset": "maveryn/cti-bench",
            "source": tsv_url,
            "sample_id": idx,
            "task_type": get_task_type(task_name),
            "prompt_mode": "original_prompt_column",
            "system_prompt": "You are a cybersecurity expert specializing in cyberthreat intelligence.",
            "generation_params": {
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "max_tokens": 2048,
            },
            "collector_scope": "raw_response_collection_only",
            "original_inference_script": "evaluation/model-prediction.ipynb",
        }

        # Preserve all original fields for downstream eval/debugging.
        metadata["original_fields"] = sample

        if task_name == "ctibench_taa":
            metadata["requires_original_eval_assets"] = True
            metadata["ground_truth_note"] = (
                "Original CTI-Bench TAA evaluation uses alias/related dictionaries; "
                "the TSV is not a simple MCQ/GT-only task."
            )

        result = {
            "task": task_name,
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": metadata,
        }
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(results)} responses to: {output_file}")
    return len(results)


def athena_eval_task_name(task_name: str) -> str:
    """Map our canonical Athena task names to AthenaBench evaluator task names."""
    mapping = {
        "athenabench_ckt": "CKT",
        "athenabench_ate": "ATE",
        "athenabench_rcm": "RCM",
        "athenabench_rms": "RMS",
        "athenabench_vsp": "VSP",
        "athenabench_taa": "TAA",
        # backward-safe
        "ckt": "CKT",
        "rms": "RMS",
        "taa": "TAA",
    }
    return mapping.get(task_name, task_name.upper())


def collect_athenabench_jsonl(task_name: str, jsonl_url: str, 
                             model, tokenizer, output_file: str, max_samples: int = None,
                             **api_kwargs):
    """Collect responses from AthenaBench GitHub JSONL tasks.

    Faithful to AthenaBench run.py:
    - use original row["prompt"] as model input
    - use original row["answer"] as ground truth
    - do not reconstruct options
    - no system prompt
    - max_new_tokens=2048
    - keep Athena-compatible response/answer fields in metadata
    """
    print(f"\n{'='*70}")
    print(f"Collecting {task_name.upper()} responses")
    print(f"JSONL source: {jsonl_url}")
    print(f"{'='*70}")
    
    dataset = load_jsonl_dataset(jsonl_url)

    if len(dataset) > 0 and "prompt" not in dataset.column_names:
        raise ValueError(
            f"AthenaBench JSONL does not contain a prompt field: {jsonl_url}. "
            f"Available columns: {dataset.column_names}"
        )

    if len(dataset) > 0 and "answer" not in dataset.column_names:
        raise ValueError(
            f"AthenaBench JSONL does not contain an answer field: {jsonl_url}. "
            f"Available columns: {dataset.column_names}"
        )
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Total samples: {len(dataset)}")
    
    results = []
    athena_task = athena_eval_task_name(task_name)

    for idx, sample in enumerate(tqdm(dataset, desc=f"Collecting {task_name.upper()}")):
        sample = dict(sample)

        # Faithful to AthenaBench run.py: row.get("prompt", "")
        prompt = sample.get("prompt", "")
        if not prompt:
            continue
        
        # Faithful to AthenaBench run.py: row.get("answer", "")
        ground_truth = str(sample.get("answer", "") or "").strip()
        
        # Faithful to models.py: no system prompt; single user prompt.
        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=2048,
            temperature=0.0,
            **api_kwargs,
        )
        
        metadata = {
            "source": jsonl_url,
            "sample_id": idx,
            "task_type": get_task_type(task_name),
            "dataset": "Athena-Software-Group/athenabench",
            "prompt_mode": "original_prompt_field",
            "answer_field": "answer",
            "generation_params": {
                "temperature": 0.0,
                "max_new_tokens": 2048,
                "system_prompt": None,
            },
            "original_inference_script": "athena_eval/run.py",
            "original_model_wrapper": "athena_eval/models.py",
            "athena_eval_task": athena_task,

            # AthenaBench run.py output compatibility:
            # run.py stores: id, prompt, response, prediction, answer.
            # We keep prediction absent here because extraction belongs to eval.
            "athena_output_compatible": {
                "id": idx,
                "prompt": prompt,
                "response": response,
                "answer": ground_truth,
            },
        }

        # Keep options in metadata only. Do not inject them into the prompt.
        choices = {}
        for letter, key in {
            "A": "option_a",
            "B": "option_b",
            "C": "option_c",
            "D": "option_d",
            "E": "option_e",
        }.items():
            if key in sample and sample.get(key):
                choices[letter] = sample.get(key)

        if choices:
            metadata["choices"] = choices
            metadata["choices_note"] = (
                "Stored for inspection only. AthenaBench inference uses original prompt field directly."
            )

        # Preserve useful original fields without changing top-level format.
        for key in [
            "id",
            "url_id",
            "url",
            "source_type",
            "processed_path",
            "raw_path",
            "char_count",
            "question_count_planned",
            "question",
            "correct_answer",
            "updated_answer",
            "explanation",
            "prompt_hash",
            "vector_score",
        ]:
            if key in sample:
                metadata[key] = sample[key]

        if task_name == "athenabench_taa":
            metadata["requires_original_eval_assets"] = True
            metadata["eval_assets"] = [
                "athena_eval/taa/aliases.csv",
                "athena_eval/taa/related_groups.csv",
            ]
            metadata["eval_note"] = (
                "AthenaBench TAA scoring uses alias and related-group dictionaries "
                "to compute correct, plausible, and combined accuracy."
            )

        if task_name == "athenabench_vsp":
            metadata["eval_note"] = (
                "AthenaBench VSP evaluation parses predicted and gold CVSS v3 vectors "
                "with CVSS3, computes MAD, then derives accuracy using the configured denominator."
            )

        # Keep the full original row for faithful downstream evaluation/debugging.
        metadata["original_fields"] = sample
        
        result = {
            "task": task_name,
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": metadata
        }
        results.append(result)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved {len(results)} responses to: {output_file}")
    return len(results)


def collect_secure_tsv(task_name: str, tsv_url: str,
                       model, tokenizer, output_file: str,
                       max_samples: int = None, **api_kwargs):
    print(f"\n{'='*70}")
    print(f"Collecting {task_name.upper()} responses")
    print(f"TSV source: {tsv_url}")
    print(f"{'='*70}")

    dataset = load_tsv_dataset(tsv_url)

    if len(dataset) > 0 and "Prompt" not in dataset.column_names:
        raise ValueError(
            f"SECURE TSV does not contain a Prompt column: {tsv_url}. "
            f"Available columns: {dataset.column_names}"
        )

    if len(dataset) > 0 and "Correct Answer" not in dataset.column_names:
        raise ValueError(
            f"SECURE TSV does not contain a Correct Answer column: {tsv_url}. "
            f"Available columns: {dataset.column_names}"
        )

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Total samples: {len(dataset)}")

    results = []

    for idx, sample in enumerate(tqdm(dataset, desc=f"Collecting {task_name.upper()}")):
        sample = dict(sample)

        # Faithful to SECURE: use Prompt column verbatim.
        prompt = sample.get("Prompt")
        if not prompt:
            continue

        # SECURE gold field.
        ground_truth = str(sample.get("Correct Answer", "") or "").strip()

        #temperature=0.7 is explicitly mentioned for SECURE in the paper.
        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=1024,
            temperature=0.7,
            **api_kwargs,
        )

        metadata = {
            "dataset": "aiforsec/SECURE",
            "source": tsv_url,
            "sample_id": idx,
            "task_type": get_task_type(task_name),
            "prompt_mode": "original_prompt_column",
            "answer_field": "Correct Answer",
            "generation_params": {
                "temperature": 0.7,
                "max_new_tokens": 1024,
            },
            "collector_scope": "raw_response_collection_only",
            "original_fields": sample,
        }

        # Preserve common SECURE fields explicitly for easier inspection.
        for key in [
            "URL",
            "Question",
            "Option A",
            "Option B",
            "Option C",
            "Option D",
            "Level",
            "Explanation",
            "CVSS v3 Vector String",
            "Overview",
            "Vulnerability",
            "Risk Evaluation",
        ]:
            if key in sample:
                metadata[key] = sample[key]

        # For MAET/CWET, the paper describes A/B/C/D/X-style prompts.
        # For KCV, the prompt is T/F/X. Do not infer type from code;
        # preserve the prompt and mark the task family.
        if task_name in {"secure_maet", "secure_cwet"}:
            metadata["official_metric"] = "accuracy"
            metadata["answer_format_note"] = "Prompt asks for A, B, C, D, or X depending on the row."
        elif task_name == "secure_kcv":
            metadata["official_metric"] = "accuracy"
            metadata["answer_format_note"] = "Prompt asks for T, F, or X."

        result = {
            "task": task_name,
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": metadata
        }
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(results)} responses to: {output_file}")
    return len(results)


def collect_seceval(model, tokenizer, output_file: str, max_samples: int = None,
                   **api_kwargs):
    """Collect responses from SecEval using official chat few-shot prompting."""
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

    results = []

    for idx, q in enumerate(tqdm(questions, desc="Collecting SecEval")):
        question = q.get("question", "")
        choices = q.get("choices", [])

        if not question or not choices:
            continue

        # Faithful to eval.py:
        # "Question: " + dataset_row["question"] + " ".join(dataset_row["choices"])
        question_text = "Question: " + question + " ".join(choices)
        question_text = question_text.replace("\n", " ")

        messages = (
            [{"role": "system", "content": instruction}]
            + chat_few_shot
            + [{"role": "user", "content": question_text}]
        )

        response = generate_response(
            model,
            tokenizer,
            messages=messages,
            max_new_tokens=5,
            **api_kwargs,
        )

        result = {
            "task": "seceval",
            "index": idx,
            "prompt": messages,
            "ground_truth": q.get("answer", ""),
            "model_response": response,
            "metadata": {
                "dataset": "XuanwuAI/SecEval",
                "source_url": dataset_url,
                "sample_id": idx,
                "question": question,
                "choices": choices,
                "id": q.get("id", ""),
                "task_type": "seceval",
                "source": q.get("source", ""),
                "topics": q.get("topics", []),
                "keyword": q.get("keyword", ""),
                "prompt_style": "official_chat_fewshot",
                "official_inference_script": "eval/eval.py",
                "generation_params": {
                    "max_new_tokens": 5,
                },
                "official_answer_extraction": (
                    "If 'Answer:' is present, remove it; then extract A-D letters, "
                    "deduplicate, sort, and join before comparing to answer."
                ),
                "original_fields": q,
            },
        }
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(results)} responses to: {output_file}")
    return len(results)


def collect_cybermetric(model, tokenizer, output_file: str, max_samples: int = None,
                        dataset_url: str = None, **api_kwargs):
    """Collect responses from CyberMetric using the original evaluator prompt."""
    print(f"\n{'='*70}")
    print("Collecting CyberMetric responses")
    print(f"{'='*70}")

    if dataset_url is None:
        dataset_url = "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-500-v1.json"

    json_data = load_json_dataset(dataset_url)
    questions_data = json_data["questions"]

    if max_samples:
        questions_data = questions_data[:max_samples]

    print(f"Total samples: {len(questions_data)}")

    system_prompt = "You are a security expert who answers questions."
    results = []

    for idx, sample in enumerate(tqdm(questions_data, desc="Collecting CYBERMETRIC")):
        question = sample["question"]
        answers = sample["answers"]
        ground_truth = str(sample["solution"]).strip()

        # Exact CyberMetric_evaluator.py prompt construction.
        options = ", ".join([f"{key}) {value}" for key, value in answers.items()])
        prompt = (
            f"Question: {question}\n"
            f"Options: {options}\n\n"
            f"Choose the correct answer (A, B, C, or D) only. "
            f"Always return in this format: 'ANSWER: X' "
        )

        response = generate_response(
            model,
            tokenizer,
            prompt,
            system_prompt=system_prompt,
            # Original CyberMetric evaluator does not specify max_tokens/temperature.
            temperature=0.0,
            max_new_tokens=1024,
            **api_kwargs,
        )

        result = {
            "task": "cybermetric",
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": {
                "dataset": "cybermetric/CyberMetric",
                "source": dataset_url,
                "sample_id": idx,
                "task_type": "mcq",
                "question": question,
                "choices": answers,
                "system_prompt": system_prompt,
                "prompt_mode": "official_evaluator_prompt",
                "official_inference_script": "CyberMetric_evaluator.py",
                "generation_params": {
                    "max_new_tokens": 1024,
                    "temperature": 0.0,
                    "temperature_note": "Not specified in original CyberMetric evaluator; set to 0.0 for deterministic raw collection.",
                    "max_tokens_note": "Not specified in original CyberMetric evaluator."
                },
                "official_answer_extraction": r"ANSWER:?\s*([A-D])",
                "official_retry_policy": "Original evaluator retries up to 5 times if regex extraction fails.",
                "original_fields": sample,
            },
        }
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(results)} responses to: {output_file}")
    return len(results)

def collect_secbench_mcq(task_name: str, dataset_url: str,
                         model, tokenizer, output_file: str,
                         max_samples: int = None, **api_kwargs):
    """Collect SecBench MCQ responses from the original released GitHub data.

    SecBench paper/repo do not provide an exact inference prompt template.
    This uses a reconstructed MCQ prompt from released fields and preserves
    the paper's exact-match MCQ scoring assumption in metadata.
    """
    print(f"\n{'='*70}")
    print(f"Collecting {task_name.upper()} responses")
    print(f"Source: {dataset_url}")
    print(f"{'='*70}")

    samples = load_concatenated_json_objects(dataset_url)

    # Match previous HF config intent: English MCQs only.
    samples = [s for s in samples if s.get("language") == "English"]

    if max_samples:
        samples = samples[:max_samples]

    print(f"Total samples: {len(samples)}")

    results = []

    for idx, sample in enumerate(tqdm(samples, desc=f"Collecting {task_name.upper()}")):
        question = str(sample.get("question", "") or "").strip()
        answers = sample.get("answers", [])
        ground_truth = str(sample.get("label", "") or "").strip()

        if not question or not answers or not ground_truth:
            continue

        prompt = (
            "Answer the following multiple-choice cybersecurity question. "
            "Select the correct option letter(s) from A, B, C, and D. "
            "Return only the letter(s), with no explanation.\n\n"
        )
        prompt += question + "\n"

        for i, answer in enumerate(answers[:4]):
            prompt += f"{chr(65 + i)}. {answer}\n"

        prompt += "Answer:"

        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=16,
            temperature=0.0,
            **api_kwargs,
        )

        result = {
            "task": task_name,
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": {
                "dataset": "secbench-git/SecBench",
                "source": dataset_url,
                "sample_id": idx,
                "task_type": "mcq",
                "prompt_mode": "reconstructed_from_released_fields",
                "official_prompt_available": False,
                "official_inference_script": None,
                "official_eval_framework": "OpenCompass",
                "official_mcq_scoring": (
                    "Exact match between model's selected letter(s) and label. "
                    "For multi-answer MCQs, incomplete or extra selections receive no credit."
                ),
                "generation_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 16,
                    "note": (
                        "SecBench paper/repo do not specify generation parameters; "
                        "capped because MCQ expects A/B/C/D letter output."
                    ),
                },
                "question": question,
                "choices": answers,
                "language": sample.get("language", ""),
                "ability": sample.get("ability", ""),
                "level": sample.get("level", ""),
                "domain": sample.get("domain", ""),
                "original_fields": sample,
            },
        }
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(results)} SecBench MCQ responses to: {output_file}")
    return len(results)
    

def collect_redsage_mcq_generation(task_name: str, dataset_name: str, subset_name: str,
                                   model, tokenizer, output_file: str,
                                   max_samples: int = None, **api_kwargs):
    """Collect RedSageMCQ responses using RedSage _em-style generation.

    Faithful pieces:
    - same RedSage cybersec_prompt_fn layout
    - include_context=False
    - generation_size=100
    - stop_sequence=["\\n"] applied after generation
    """
    print(f"\n{'='*70}")
    print(f"Collecting {task_name.upper()} RedSageMCQ responses")
    print(f"Dataset: {dataset_name}/{subset_name}")
    print(f"{'='*70}")

    dataset = load_dataset(dataset_name, subset_name, split="test")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Total samples: {len(dataset)}")

    results = []

    for idx, sample in enumerate(tqdm(dataset, desc=f"Collecting {task_name.upper()}")):
        sample = dict(sample)

        prompt = build_redsage_prompt(sample, include_context=False)
        ground_truth = redsage_answer_letter(sample["solution"])

        raw_response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=100,
            temperature=0.0,
            **api_kwargs,
        )

        response = apply_stop_sequence(raw_response, stop_sequence=["\n"])

        result = {
            "task": task_name,
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": {
                "dataset": dataset_name,
                "subset": subset_name,
                "sample_id": idx,
                "task_type": "mcq",
                "prompt_mode": "redsage_cybersec_prompt_fn_include_context_false",
                "official_inference_script": "RISys-Lab/RedSage eval/cybersecurity_benchmarks.py",
                "official_prompt_function": "cybersec_prompt_fn",
                "include_context": False,
                "collector_mode": "generative_em_style",
                "official_default_metric": "loglikelihood_acc",
                "official_em_metrics": [
                    "exact_match",
                    "prefix_exact_match",
                    "regex_mcq_acc"
                ],
                "generation_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 100,
                    "official_em_generation_size": 100,
                    "official_em_stop_sequence": ["\\n"],
                    "stop_sequence_applied_post_generation": True,
                },
                "raw_model_response_before_stop": raw_response,
                "question": sample.get("question", ""),
                "choices": sample.get("answers", {}),
                "content_preserved_but_not_prompted": sample.get("content", ""),
                "original_fields": sample,
            },
        }
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(results)} RedSageMCQ responses to: {output_file}")
    return len(results)


def collect_cissp(model, tokenizer, output_file: str, dataset_path: str = None,
                 max_samples: int = None, **api_kwargs):
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
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        
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


def is_non_chinese_sample(text: str) -> bool:
    """Check if text contains no Chinese characters.
    
    Note: This only filters out Chinese text; it does not verify the text is English.
    Used for SEvenLLM-Bench which is specifically bilingual (EN/ZH).
    """
    import re
    # Chinese Unicode ranges: CJK Unified Ideographs
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return not bool(chinese_pattern.search(text))


def collect_sevenllm(model, tokenizer, output_file: str, max_samples: int = None, **api_kwargs):
    """Collect responses from SEvenLLM-Bench (English samples only).
    
    SEvenLLM-Bench is a structured extraction benchmark with 24 cybersecurity task categories.
    This function filters to English-only samples (650 out of 1300 total) based on the input field.
    
    Dataset: Multilingual-Multimodal-NLP/SEVENLLM-Dataset on HuggingFace
    """
    from huggingface_hub import hf_hub_download
    
    print(f"\n{'='*70}")
    print("Collecting SEVENLLM responses (English filtered)")
    print("Dataset: Multilingual-Multimodal-NLP/SEVENLLM-Dataset")
    print(f"{'='*70}")
    
    # Download and load dataset manually (HF auto-loader fails due to mixed output types)
    file_path = hf_hub_download(
        repo_id='Multilingual-Multimodal-NLP/SEVENLLM-Dataset',
        filename='test.jsonl',
        repo_type='dataset'
    )
    
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Total samples in dataset: {len(samples)}")
    
    # Filter to non-Chinese samples based on INPUT field (not instruction)
    # The dataset has 650 samples with English input and 650 with Chinese input
    english_samples = [s for s in samples if is_non_chinese_sample(s.get('input', ''))]
    
    print(f"Non-Chinese samples: {len(english_samples)}")
    
    if max_samples:
        english_samples = english_samples[:max_samples]
    
    print(f"Samples to process: {len(english_samples)}")
    
    # Collect responses
    results = []
    for idx, sample in enumerate(tqdm(english_samples, desc="Collecting SEVENLLM")):
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        ground_truth = sample.get('output', '')
        category = sample.get('category', 'unknown')
        
        # Format prompt using SEvenLLM's native format: instruction + input
        # This matches their training and evaluation approach
        # Format prompt to mirror the original SEvenLLM inference script
        tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).lower()
        if "qwen" in tokenizer_name:
            prompt = SEVENLLM_PROMPT_DICT["prompt_input_qwen"].format(
                instruction=instruction,
                input=input_text,
            )
        else:
            prompt = SEVENLLM_PROMPT_DICT["prompt_input"].format(
                instruction=instruction,
                input=input_text,
            )

        # Generate response
        response = generate_response(model, tokenizer, prompt, max_new_tokens=2048, task_name="sevenllm", **api_kwargs)
        
        # Handle ground_truth - it may be dict or string
        if isinstance(ground_truth, dict):
            ground_truth = json.dumps(ground_truth)
        else:
            ground_truth = str(ground_truth)
        
        # Store raw result
        # Include input_text in metadata for SEvenLLM's evaluation prompt format
        result = {
            "task": "sevenllm",
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": {
                "dataset": "Multilingual-Multimodal-NLP/SEVENLLM-Dataset",
                "category": category,
                "instruction": instruction,
                "input": input_text,  # Cybersecurity incident content for judge context
                "task_type": "sevenllm"  # Structured JSON extraction
            }
        }
        results.append(result)
    
    # Save to JSONL
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"✓ Saved {len(results)} responses to: {output_file}")
    
    # Print category distribution
    category_counts = {}
    for r in results:
        cat = r['metadata']['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    print(f"\nCategory distribution ({len(category_counts)} categories):")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")
    if len(category_counts) > 10:
        print(f"  ... and {len(category_counts) - 10} more categories")
    
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
    
    # vLLM options for fast batch inference
    parser.add_argument("--use_vllm", action="store_true", 
                       help="Use vLLM for fast batch inference (requires vllm package)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory fraction to use with vLLM (0.0-1.0, default: 0.9)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for vLLM inference (default: 16)")
    
    # Collection options
    parser.add_argument("--tasks", nargs="+", default=["mcq", "rcm", "vsp", "ate"], 
                       help="Tasks to collect: mcq, rcm, vsp, ate, cti_taa, ckt, rms, taa, mmlu-cs, secure_maet, secure_cwet, secure_kcv, secbench, redsage_frameworks, redsage_generals, redsage_skills, redsage_cli, redsage_kali, cybermetric, seceval, cissp, sevenllm. Note: cti_taa is CTI-Bench TAA (50 items), taa is AthenaBench TAA (100 items), sevenllm is SEvenLLM-Bench English samples (650 items)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for JSONL files")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples per task (for testing)")
    
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
            args.vllm_gpu_memory_utilization
        )
        model = None
        tokenizer = None
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
    
    with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Prepare inference kwargs (api_kwargs name kept for backward compatibility)
    api_kwargs = {
        'use_api': args.use_api,
        'use_vllm': args.use_vllm,
        'vllm_model': vllm_model,
        'batch_size': args.batch_size,
        'api_endpoint': args.api_endpoint,
        'api_model': args.api_model,
        'api_key': args.api_key
    }
    
    # Task configurations
    # Format: "task_name": (dataset_type, dataset_name, subset_or_url)
    task_map = {
        # CTI-Bench original TSV files
        "mcq": ("ctibench_tsv", "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-mcq.tsv", "ctibench_mcq"),
        "rcm": ("ctibench_tsv", "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-rcm.tsv", "ctibench_rcm"),
        "rcm_2021": ("ctibench_tsv", "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-rcm-2021.tsv", "ctibench_rcm_2021"),
        "vsp": ("ctibench_tsv", "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-vsp.tsv", "ctibench_vsp"),
        "ate": ("ctibench_tsv", "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-ate.tsv", "ctibench_ate"),
        "cti_taa": ("ctibench_tsv", "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-taa.tsv", "ctibench_taa"),

        # AthenaBench GitHub JSONL tasks
        "ckt": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-ckt-3k.jsonl", "athenabench_ckt"),
        "ate_athena": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-ate.jsonl", "athenabench_ate"),
        "rcm_athena": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-rcm.jsonl", "athenabench_rcm"),
        "rms": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-rms.jsonl", "athenabench_rms"),
        "vsp_athena": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-vsp.jsonl", "athenabench_vsp"),
        "taa": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-taa.jsonl", "athenabench_taa"),

        # Other HuggingFace benchmarks
        "mmlu-cs": ("mmlu_generation", "cais/mmlu", "computer_security"),
        "mmlu-cs-logprobs": ("mmlu_logprobs", "cais/mmlu", "computer_security"),    
        "secure_maet": ("secure_tsv", "https://raw.githubusercontent.com/aiforsec/SECURE/main/Dataset/SECURE%20-%20MAET.tsv", "secure_maet"),
        "secure_cwet": ("secure_tsv", "https://raw.githubusercontent.com/aiforsec/SECURE/main/Dataset/SECURE%20-%20CWET.tsv", "secure_cwet"),
        "secure_kcv": ("secure_tsv", "https://raw.githubusercontent.com/aiforsec/SECURE/main/Dataset/SECURE%20-%20KCV.tsv", "secure_kcv"),
        "secbench": ("secbench_mcq", "https://raw.githubusercontent.com/secbench-git/SecBench/main/data/MCQs_2730.jsonl", "secbench"),

        # RedSageMCQ
        "redsage_frameworks": ("redsage_generation", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_knowledge_frameworks"),
        "redsage_generals": ("redsage_generation", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_knowledge_generals"),
        "redsage_skills": ("redsage_generation", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_skills"),
        "redsage_cli": ("redsage_generation", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_tools_cli"),
        "redsage_kali": ("redsage_generation", "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_tools_kali"),
    }
    # Collect responses for each task
    summary = {}
    
    for task_name in args.tasks:
        output_file = os.path.join(args.output_dir, f"{task_name}_responses.jsonl")
        
        try:
            if task_name in task_map:
                # Handle mapped tasks based on type
                task_type, source, subset_or_url = task_map[task_name]
                if task_type == "ctibench_tsv":
                    count = collect_ctibench_tsv(
                        subset_or_url,
                        source,
                        model,
                        tokenizer,
                        output_file,
                        args.max_samples,
                        **api_kwargs,
                    )

                elif task_type == "secure_tsv":
                    count = collect_secure_tsv(
                        subset_or_url,
                        source,
                        model,
                        tokenizer,
                        output_file,
                        args.max_samples,
                        **api_kwargs,
                    )
                    
                elif task_type == "mmlu_generation":
                    count = collect_mmlu_generation(
                        task_name,
                        source,
                        subset_or_url,
                        model,
                        tokenizer,
                        output_file,
                        args.max_samples,
                        **api_kwargs,
                    )

                elif task_type == "mmlu_logprobs":
                    count = collect_mmlu_logprobs(
                        task_name,
                        source,
                        subset_or_url,
                        model,
                        tokenizer,
                        output_file,
                        args.max_samples,
                        **api_kwargs,
                    )
                elif task_type == "secbench_mcq":
                    count = collect_secbench_mcq(
                        subset_or_url,
                        source,
                        model,
                        tokenizer,
                        output_file,
                        args.max_samples,
                        **api_kwargs,
                    )

                elif task_type == "redsage_generation":
                    count = collect_redsage_mcq_generation(
                        task_name,
                        source,
                        subset_or_url,
                        model,
                        tokenizer,
                        output_file,
                        args.max_samples,
                        **api_kwargs,
                    )

                elif task_type == "hf":
                    # HuggingFace datasets.
                    # subset_or_url stores the HF subset/config name.
                    count = collect_huggingface_benchmark(
                        task_name,
                        source,
                        subset_or_url,
                        model,
                        tokenizer,
                        output_file,
                        args.max_samples,
                        **api_kwargs,
                    )

                elif task_type == "jsonl":
                    # GitHub JSONL datasets.
                    # subset_or_url stores the canonical task name, e.g. athenabench_ckt.
                    count = collect_athenabench_jsonl(
                        subset_or_url,
                        source,
                        model,
                        tokenizer,
                        output_file,
                        args.max_samples,
                        **api_kwargs,
                    )
    
            elif task_name == "seceval":
                count = collect_seceval(model, tokenizer, output_file, 
                                       args.max_samples, **api_kwargs)
            elif task_name == "cybermetric":
                count = collect_cybermetric(model, tokenizer, output_file,
                                        args.max_samples, **api_kwargs)
            elif task_name == "cissp":
                count = collect_cissp(model, tokenizer, output_file, args.cissp_path,
                                     args.max_samples, **api_kwargs)
            elif task_name == "sevenllm":
                count = collect_sevenllm(model, tokenizer, output_file,
                                        args.max_samples, **api_kwargs)
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
