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
        "rcm": "rcm",  # CWE ID extraction
        "vsp": "vsp",  # CVSS vector extraction
        "ate": "ate",  # MITRE ATT&CK techniques
        "rms": "rms",  # Risk mitigation strategies
        "taa": "taa",  # Threat actor attribution (AthenaBench)
        "cti_taa": "taa",  # Threat actor attribution (CTI-Bench)
        # SEvenLLM-Bench tasks (structured JSON extraction)
        "sevenllm": "sevenllm",  # Multi-category CTI extraction tasks
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
        "temperature": temperature
    }
    
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
                     messages: list = None, **kwargs) -> str:
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
    """Collect responses from HuggingFace datasets (CTI-Bench, MMLU-CS, SECURE, SecBench, RedSageMCQ)"""
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


def collect_athenabench_jsonl(task_name: str, jsonl_url: str, 
                             model, tokenizer, output_file: str, max_samples: int = None,
                             **api_kwargs):
    """Collect responses from AthenaBench GitHub JSONL tasks (CKT, RMS, TAA)"""
    print(f"\n{'='*70}")
    print(f"Collecting {task_name.upper()} responses")
    print(f"JSONL source: {jsonl_url}")
    print(f"{'='*70}")
    
    # Load JSONL dataset
    dataset = load_jsonl_dataset(jsonl_url)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Total samples: {len(dataset)}")
    
    # Collect responses
    results = []
    for idx, sample in enumerate(tqdm(dataset, desc=f"Collecting {task_name.upper()}")):
        # AthenaBench JSONL format has 'question', 'prompt', and 'answer' fields
        question = sample.get('question', '')
        prompt = sample.get('prompt', question)
        
        # Get ground truth answer
        ground_truth = sample.get('answer', sample.get('correct_answer', '')).strip()
        
        # Generate response
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        
        # Prepare metadata with options if available (for CKT)
        metadata = {
            "source": jsonl_url,
            "sample_id": idx,
            "task_type": get_task_type(task_name)  # Add task type for judge
        }
        
        # Add options for CKT (5-option MCQ)
        if 'option_a' in sample:
            metadata['choices'] = {
                'A': sample.get('option_a', ''),
                'B': sample.get('option_b', ''),
                'C': sample.get('option_c', ''),
                'D': sample.get('option_d', ''),
                'E': sample.get('option_e', '')
            }
        
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


def collect_seceval(model, tokenizer, output_file: str, max_samples: int = None,
                   **api_kwargs):
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

    results = []
    for idx, q in enumerate(tqdm(questions, desc="Collecting SecEval")):
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

        response = generate_response(
            model,
            tokenizer,
            messages=messages,
            max_new_tokens=1024,
            **api_kwargs,
        )

        result = {
            "task": "seceval",
            "index": idx,
            "prompt": messages,
            "ground_truth": q.get("answer", ""),
            "model_response": response,
            "metadata": {
                "question": question,
                "choices": choices,
                "id": q.get("id", ""),
                "task_type": "seceval",
                "source": q.get("source", ""),
                "topics": q.get("topics", []),
                "keyword": q.get("keyword", ""),
                "prompt_style": "chat_fewshot",
            },
        }
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(results)} responses to: {output_file}")
    return len(results)


def collect_cybermetric(model, tokenizer, output_file: str, max_samples: int = None,
                        **api_kwargs):
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
    results = []

    for idx, sample in enumerate(tqdm(dataset, desc="Collecting CYBERMETRIC")):
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

        prompt = (
            f"Question: {question}\n"
            f"Options: {options}\n\n"
            f"Choose the correct answer (A, B, C, or D) only. "
            f"Always return in this format: 'ANSWER: X'"
        )

        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=1024,
            system_prompt=system_prompt,
            **api_kwargs
        )

        result = {
            "task": "cybermetric",
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_response": response,
            "metadata": {
                "dataset": "RISys-Lab/Benchmarks_CyberSec_CyberMetrics",
                "subset": "cyberMetric_500",
                "sample_id": idx,
                "task_type": "mcq",
                "question": question,
                "choices": answers,
            }
        }
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(results)} responses to: {output_file}")
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
    print("Collecting SEVENLLM responses (English only)")
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
        prompt = f"{instruction}\n\n{input_text}"
        
        # Generate response
        response = generate_response(model, tokenizer, prompt, max_new_tokens=2048, **api_kwargs)
        
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
        # RISys-Lab CTI-Bench (HuggingFace)
        "mcq": ("hf", "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-mcq"),
        "rcm": ("hf", "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-rcm"),
        "vsp": ("hf", "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-vsp"),
        "ate": ("hf", "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-ate"),
        "cti_taa": ("hf", "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-taa"),  # Original TAA: 50 items
        
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
    }
    
    # Collect responses for each task
    summary = {}
    
    for task_name in args.tasks:
        output_file = os.path.join(args.output_dir, f"{task_name}_responses.jsonl")
        
        try:
            if task_name in task_map:
                # Handle mapped tasks based on type
                task_type, source, subset_or_url = task_map[task_name]
                
                if task_type == "hf":
                    # HuggingFace datasets (CTI-Bench, MMLU-CS, SECURE, SecBench)
                    count = collect_huggingface_benchmark(task_name, source, subset_or_url, 
                                                          model, tokenizer, output_file, 
                                                          args.max_samples, **api_kwargs)
                elif task_type == "jsonl":
                    # AthenaBench GitHub JSONL tasks (CKT, RMS, TAA)
                    count = collect_athenabench_jsonl(task_name, source, 
                                                      model, tokenizer, output_file, 
                                                      args.max_samples, **api_kwargs)
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
