#!/usr/bin/env python3
"""
Evaluate fine-tuned model on multiple cybersecurity benchmarks

Supported benchmarks:
- CTI-Bench: MCQ, RCM, VSP, ATE (4 tasks)
- CyberMetric-500: Cybersecurity knowledge questions
- SecEval: 2126 cybersecurity knowledge questions
- CISSP: Cybersecurity certification questions

Supports both local model inference and API endpoint evaluation.
"""

import os
import re
import json
import yaml
import time
import torch
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional peft import for LoRA adapter support
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_path: str, base_model: str = None, is_base: bool = False):
    """Load model (base or fine-tuned) with optional LoRA adapters
    
    Args:
        model_path: Path to model checkpoint
        base_model: Base model name (required if model_path contains LoRA adapters)
        is_base: If True, load as base model without any adapters
    """
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
        # Load base model only (pre-training evaluation)
        model = AutoModelForCausalLM.from_pretrained(
            model_path_to_load,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    elif os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # LoRA checkpoint (post-training evaluation)
        print("Loading LoRA adapters...")
        if not base_model:
            raise ValueError("base_model must be specified when loading LoRA adapters")
        if not HAS_PEFT:
            raise ImportError(
                "PEFT library not found but LoRA adapter detected. "
                "Install peft with: pip install peft"
            )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        # Full merged model
        model = AutoModelForCausalLM.from_pretrained(
            model_path_to_load,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    return model, tokenizer


def chat_completion_api(endpoint: str, model_name: str, prompt: str,
                       api_key: str = "", max_tokens: int = 1024,
                       temperature: float = 0.1, retries: int = 3,
                       api_version: str = "2024-12-01-preview",
                       messages: list = None) -> str:
    """Call Azure OpenAI chat completions API using the official SDK."""
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )

    if messages is None:
        messages = [{"role": "user", "content": prompt}]

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
                time.sleep(2 ** attempt)
            else:
                print(f"API call failed after {retries} attempts: {e}")
    return "ERROR: exhausted retries"


# Kept for backward compatibility — delegates to chat_completion_api
def chat_responses_api(endpoint: str, model_name: str, prompt: str,
                      api_key: str = "", max_tokens: int = 1024,
                      temperature: float = 0.0, retries: int = 3) -> str:
    return chat_completion_api(endpoint, model_name, prompt,
                               api_key=api_key, max_tokens=max_tokens,
                               temperature=temperature, retries=retries)


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 1024,
                     use_api: bool = False, api_endpoint: str = None,
                     api_model: str = None, api_key: str = "", api_version: str = None,
                     api_style: str = "chat_completions", **kwargs):
    """Generate response from model (local or API)"""
    
    # API mode
    if use_api:
        if not api_endpoint or not api_model:
            raise ValueError("API endpoint and model name required for API mode")

        # Get API version from parameter or environment variable
        if api_version is None:
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        return chat_completion_api(api_endpoint, api_model, prompt, api_key, max_new_tokens,
                                   api_version=api_version)
    
    # Local model mode
    # Always try to use tokenizer's native chat template first
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        # Fallback to model-specific templates
        model_name = tokenizer.name_or_path.lower()
        
        if "gemma" in model_name:
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif "llama" in model_name:
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def extract_final_answer(text: str, task_type: str = None) -> str:
    """Extract final answer from model response
    
    For MCQ: Looks for A/B/C/D letters, preferring explicit answer patterns
    For other tasks: Returns last non-empty line
    """
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    
    if task_type == "mcq":
        # For MCQ, look for A, B, C, or D with specific priority
        
        # Priority 1: "Answer: X" or "Final Answer: X" patterns (most reliable)
        answer_patterns = [
            r'\*\*(?:Final )?Answer:\*\*\s*([A-D])\b',  # **Answer:** B or **Final Answer:** B
            r'(?:Final )?Answer:\s*([A-D])\b',           # Answer: B or Final Answer: B
        ]
        for pattern in answer_patterns:
            for line in reversed(lines):
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1).upper()
        
        # Priority 2: Single letter lines (exact match)
        for line in reversed(lines):
            if line.upper() in ['A', 'B', 'C', 'D']:
                return line.upper()
        
        # Priority 3: Lines starting with the letter (not followed by ")")
        for line in reversed(lines):
            first_char = line[0].upper() if line else ''
            if first_char in ['A', 'B', 'C', 'D'] and (len(line) == 1 or line[1] != ')'):
                return first_char
        
        # Priority 4: Any occurrence of A/B/C/D in reverse order
        mcq_pattern = r'\b([A-D])\b'
        for line in reversed(lines):
            match = re.search(mcq_pattern, line.upper())
            if match:
                return match.group(1)
    
    # Default: return last line
    return lines[-1] if lines else text


def parse_ids_from_text(text: str) -> set:
    """Extract MITRE technique IDs or CWE IDs from text"""
    # Pattern for MITRE IDs: T1234 or T1234.567
    # Pattern for CWE IDs: CWE-123
    pattern = r'\b(?:T\d{4}(?:\.\d{3})?|CWE-\d+)\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return set([m.upper() for m in matches])


def compute_set_metrics(pred_set: set, gold_set: set) -> dict:
    """Compute precision, recall, F1 for set-based predictions"""
    if not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": len(pred_set) == 0}
    
    true_positives = len(pred_set & gold_set)
    
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(gold_set) if gold_set else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = pred_set == gold_set
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match
    }


def evaluate_mcq(model, tokenizer, dataset, detailed_output=None, **api_kwargs):
    """Evaluate on CTI-MCQ (multiple choice questions)"""
    print("\n" + "="*50)
    print("Evaluating CTI-MCQ (Multiple Choice Questions)")
    print("="*50)
    
    correct = 0
    total = 0
    detailed_results = []
    
    for idx, sample in enumerate(tqdm(dataset, desc="CTI-MCQ")):
        prompt = sample['Prompt']
        ground_truth = sample.get('GT', '').strip()
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        answer = extract_final_answer(response, task_type="mcq")
        
        # Exact match on normalized answers to avoid substring-based false positives
        is_correct = answer.strip().lower() == ground_truth.strip().lower()
        if is_correct:
            correct += 1
        
        total += 1
        
        # Save detailed result
        if detailed_output is not None:
            detailed_results.append({
                "index": idx,
                "prompt": prompt,
                "llm_output": response,
                "pred": answer,
                "gold": ground_truth,
                "correct": is_correct
            })
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save detailed results
    if detailed_output is not None:
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"Detailed results saved to: {detailed_output}")
    
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_rcm(model, tokenizer, dataset, detailed_output=None, **api_kwargs):
    """Evaluate on CTI-RCM (CVE to CWE mapping)"""
    print("\n" + "="*50)
    print("Evaluating CTI-RCM (CVE to CWE Mapping)")
    print("="*50)
    
    correct = 0
    total = 0
    detailed_results = []
    
    for idx, sample in enumerate(tqdm(dataset, desc="CTI-RCM")):
        prompt = sample['Prompt']
        ground_truth = sample.get('GT', '').strip()
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        answer = extract_final_answer(response)
        
        # Extract CWE IDs
        pred_cwes = parse_ids_from_text(answer)
        gold_cwes = parse_ids_from_text(ground_truth)
        
        # Accuracy: exact match of CWE sets
        is_correct = pred_cwes == gold_cwes
        if is_correct:
            correct += 1
        
        total += 1
        
        # Save detailed result
        if detailed_output is not None:
            detailed_results.append({
                "index": idx,
                "prompt": prompt,
                "llm_output": response,
                "pred": sorted(list(pred_cwes)),
                "gold": sorted(list(gold_cwes)),
                "correct": is_correct
            })
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save detailed results
    if detailed_output is not None:
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"Detailed results saved to: {detailed_output}")
    
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_vsp(model, tokenizer, dataset, detailed_output=None, **api_kwargs):
    """Evaluate on CTI-VSP (CVSS score prediction)"""
    print("\n" + "="*50)
    print("Evaluating CTI-VSP (CVSS Score Prediction)")
    print("="*50)
    
    errors = []
    detailed_results = []
    
    for idx, sample in enumerate(tqdm(dataset, desc="CTI-VSP")):
        prompt = sample['Prompt']
        ground_truth = sample.get('GT', '').strip()
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        answer = extract_final_answer(response)
        
        # Extract numeric CVSS score
        try:
            # Try to find a number in ground truth
            gold_score = float(re.search(r'\d+\.?\d*', ground_truth).group())
            pred_score = float(re.search(r'\d+\.?\d*', answer).group())
            
            error = abs(pred_score - gold_score)
            errors.append(error)
            
            # Save detailed result
            if detailed_output is not None:
                detailed_results.append({
                    "index": idx,
                    "prompt": prompt,
                    "llm_output": response,
                    "pred": pred_score,
                    "gold": gold_score,
                    "error": error
                })
        except (AttributeError, ValueError):
            # If parsing fails, count as maximum error
            errors.append(10.0)  # CVSS max is 10
            if detailed_output is not None:
                detailed_results.append({
                    "index": idx,
                    "prompt": prompt,
                    "llm_output": response,
                    "pred": None,
                    "gold": ground_truth,
                    "error": 10.0,
                    "parse_error": True
                })
    
    mad = np.mean(errors) if errors else 0.0  # Mean Absolute Deviation (same as MAE)
    
    print(f"\nMAD (Mean Absolute Deviation): {mad:.4f}")
    
    # Save detailed results
    if detailed_output is not None:
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"Detailed results saved to: {detailed_output}")
    
    return {"mad": mad, "total": len(errors)}


def evaluate_ate(model, tokenizer, dataset, detailed_output=None, **api_kwargs):
    """Evaluate on CTI-ATE (MITRE ATT&CK Technique Extraction)
    
    Uses micro-averaging: pool all predictions across dataset for P/R/F1.
    This gives equal weight to each technique (standard for extraction tasks).
    """
    print("\n" + "="*50)
    print("Evaluating CTI-ATE (Attack Technique Extraction)")
    print("="*50)
    
    # Micro-averaging: global counters
    tp_total = 0
    fp_total = 0
    fn_total = 0
    exact_matches = 0
    detailed_results = []
    
    for idx, sample in enumerate(tqdm(dataset, desc="CTI-ATE")):
        prompt = sample['Prompt']
        ground_truth = sample.get('GT', '').strip()
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        answer = extract_final_answer(response)
        
        # Extract MITRE technique IDs
        pred_techniques = parse_ids_from_text(answer)
        gold_techniques = parse_ids_from_text(ground_truth)
        
        # Compute per-sample TP, FP, FN
        tp = len(pred_techniques & gold_techniques)
        fp = len(pred_techniques - gold_techniques)
        fn = len(gold_techniques - pred_techniques)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        if pred_techniques == gold_techniques:
            exact_matches += 1
        
        # Save detailed result
        if detailed_output is not None:
            detailed_results.append({
                "index": idx,
                "prompt": prompt,
                "llm_output": response,
                "pred": sorted(list(pred_techniques)),
                "gold": sorted(list(gold_techniques)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "exact_match": pred_techniques == gold_techniques
            })
    
    # Compute micro-averaged metrics
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match_rate = exact_matches / len(dataset) if len(dataset) > 0 else 0.0
    
    print(f"\nMicro-averaged metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Exact Match: {exact_match_rate:.4f}")
    print(f"Total TP/FP/FN: {tp_total}/{fp_total}/{fn_total}")
    
    # Save detailed results
    if detailed_output is not None:
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"Detailed results saved to: {detailed_output}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match_rate,
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total
    }


def evaluate_cybermetric(model, tokenizer, dataset_path: str = None, detailed_output=None, **api_kwargs):
    """Evaluate on CyberMetric-500 benchmark
    
    Args:
        dataset_path: Path to CyberMetric-500-v1.json file or URL
    """
    print("\n" + "="*50)
    print("Evaluating CyberMetric-500 (Cybersecurity Knowledge)")
    print("="*50)
    
    # Load CyberMetric dataset
    if dataset_path is None:
        dataset_path = "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-500-v1.json"
    
    if dataset_path.startswith('http'):
        response = requests.get(dataset_path)
        data = response.json()
    else:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    
    questions = data['questions']
    correct = 0
    total = 0
    detailed_results = []
    
    for idx, item in enumerate(tqdm(questions, desc="CyberMetric-500")):
        question = item['question']
        answers = item['answers']  # Dict like {"A": "...", "B": "...", ...}
        correct_answer = item['solution']
        
        # Format options (official CyberMetric format)
        options_str = ', '.join([f"{key}) {value}" for key, value in answers.items()])
        
        # Official CyberMetric prompt format
        # System: "You are a security expert who answers questions."
        # User: "Question: {question}\nOptions: {options}\n\nChoose the correct answer (A, B, C, or D) only. Always return in this format: 'ANSWER: X' "
        prompt = f"Question: {question}\nOptions: {options_str}\n\nChoose the correct answer (A, B, C, or D) only. Always return in this format: 'ANSWER: X' "
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        
        # Extract using official CyberMetric pattern: ANSWER:? [A-D]
        match = re.search(r"ANSWER:?\s*([A-D])", response, re.IGNORECASE)
        answer = match.group(1).upper() if match else extract_final_answer(response, task_type="mcq")
        
        is_correct = answer.upper() == correct_answer.upper()
        if is_correct:
            correct += 1
        
        total += 1
        
        # Save detailed result
        if detailed_output is not None:
            detailed_results.append({
                "index": idx,
                "question": question,
                "answers": answers,
                "llm_output": response,
                "pred": answer,
                "gold": correct_answer,
                "correct": is_correct
            })
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save detailed results
    if detailed_output is not None:
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"Detailed results saved to: {detailed_output}")
    
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_cissp(model, tokenizer, dataset_path: str = None, detailed_output=None, **api_kwargs):
    """Evaluate on CISSP benchmark (cybersecurity certification questions)
    
    Args:
        dataset_path: Path to CISSP JSON file (list of questions with A-D choices)
    """
    print("\n" + "="*50)
    print("Evaluating CISSP (Cybersecurity Certification)")
    print("="*50)
    
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
    
    correct = 0
    total = 0
    detailed_results = []
    
    for idx, item in enumerate(tqdm(questions, desc="CISSP")):
        # Extract question and choices (flexible key names)
        question = item.get('question') or item.get('Prompt') or ""
        
        # Handle different choice formats
        choices = {}
        if isinstance(item.get('answers'), dict):
            choices = item['answers']
        elif isinstance(item.get('options'), dict):
            choices = item['options']
        elif isinstance(item.get('choices'), list):
            # Convert list to dict with A-D labels
            labels = ['A', 'B', 'C', 'D']
            for label, choice in zip(labels, item['choices']):
                choices[label] = choice
        else:
            # Try direct A/B/C/D keys
            for label in ['A', 'B', 'C', 'D']:
                if label in item:
                    choices[label] = item[label]
        
        # Get correct answer
        correct_answer = ""
        for key in ['solution', 'answer', 'GT', 'correct_answer']:
            if key in item:
                ans = str(item[key]).strip().upper()
                # Extract just the letter
                match = re.search(r'[A-D]', ans)
                if match:
                    correct_answer = match.group(0)
                    break
        
        if not question or not choices or not correct_answer:
            continue
        
        # Format prompt - single answer MCQ format
        options_str = '\n'.join([f"{key}: {value}" for key, value in sorted(choices.items())])
        prompt = f"Question:\n{question}\n\n{options_str}\n\nChoose the single best answer and reply with ONLY one letter: A, B, C, or D."
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        
        # Extract answer (look for A-D in response)
        answer = extract_final_answer(response, task_type="mcq")
        
        is_correct = answer.upper() == correct_answer.upper()
        if is_correct:
            correct += 1
        
        total += 1
        
        # Save detailed result
        if detailed_output is not None:
            detailed_results.append({
                "index": idx,
                "question": question,
                "choices": choices,
                "llm_output": response,
                "pred": answer,
                "gold": correct_answer,
                "correct": is_correct
            })
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save detailed results
    if detailed_output is not None:
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"Detailed results saved to: {detailed_output}")
    
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_seceval(model, tokenizer, dataset_path: str = None, detailed_output=None, **api_kwargs):
    """Evaluate on SecEval benchmark (2126 cybersecurity knowledge questions)
    
    Note: GitHub docs mention 2000 questions, but the actual dataset contains 2126 questions
    
    Args:
        dataset_path: Path to questions.json file or URL
    """
    print("\n" + "="*50)
    print("Evaluating SecEval (Cybersecurity Knowledge)")
    print("="*50)
    
    # Load SecEval dataset
    if dataset_path is None:
        dataset_path = "https://huggingface.co/datasets/XuanwuAI/SecEval/resolve/main/questions.json"
    
    if dataset_path.startswith('http'):
        response = requests.get(dataset_path)
        data = response.json()
    else:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    
    # Official SecEval instruction and few-shot examples
    instruction = "Below are multiple-choice questions concerning cybersecurity. Please select the correct answers and respond with the letters ABCD only."
    
    # Official SecEval few-shot examples
    few_shot_examples = """Question: You are reviewing a piece of code to identify potential vulnerabilities related to NULL pointer dereferences. Which of the following strategies should you consider implementing to mitigate the risk? Choose all that apply. A: Use a programming language that inherently manages pointers and prevents direct memory access. B: Apply input validation to all variables and data stores that receive data from external sources to ensure they are initialized to expected values. C: Explicitly initialize all variables and other data stores, either during declaration or just before the first usage. D: Implement a custom memory management system that automatically assigns a unique identifier to each pointer.
Answer: ABC

Question: Analyze the code snippet provided. Which of the following practices should be implemented in a multithreaded application to prevent NULL pointer dereference vulnerabilities? Select all that apply. A: Ensure that proper locking APIs are used to lock the pointer before the if statement and unlock when it has finished to prevent race conditions. B: Check the return value of all functions that return a value, including network-related functions, and verify that it is non-null before using it. C: Use automated static analysis tools that target this type of weakness, understanding that while not perfect, they can still be effective. D: Verify that a non-nil 'response' is present before deferring 'response.Body.Close()' to handle cases where the 'Do' method returns an error.
Answer: ABCD

"""
    
    correct = 0
    total = 0
    correct_by_topic = {}
    total_by_topic = {}
    detailed_results = []
    
    for idx, item in enumerate(tqdm(data, desc="SecEval")):
        question = item['question']
        choices = item['choices']  # List of choices
        correct_answer = item['answer']  # String like "ABCD" or "AB" (multi-select)
        topics = item.get('topics', ['Unknown'])
        
        # Official SecEval question format: "Question: " + question + " ".join(choices)
        question_text = "Question: " + question + " " + " ".join(choices)
        question_text = question_text.replace("\n", " ")  # Remove newlines as in official script
        
        # Build full prompt with instruction + few-shot + question
        prompt = instruction + "\n\n" + few_shot_examples + question_text + "\n"
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        
        # Official SecEval extraction logic (from eval.py line 147-153)
        # Strip "Answer:" prefix if present
        llm_output = response
        if "Answer:" in llm_output:
            llm_output = llm_output.replace("Answer:", "")
        # Extract sorted unique letters A-D
        llm_answer = "".join(sorted(list(set(re.findall(r"[A-D]", llm_output)))))
        
        # Normalize correct answer
        correct_answer_normalized = "".join(sorted(correct_answer.upper()))
        
        is_correct = (llm_answer.lower() == correct_answer_normalized.lower())
        
        if is_correct:
            correct += 1
        
        # Track by topic
        for topic in topics:
            if topic not in correct_by_topic:
                correct_by_topic[topic] = 0
                total_by_topic[topic] = 0
            
            if is_correct:
                correct_by_topic[topic] += 1
            total_by_topic[topic] += 1
        
        total += 1
        
        # Save detailed result
        if detailed_output is not None:
            detailed_results.append({
                "index": idx,
                "question": question,
                "choices": choices,
                "llm_output": response,
                "pred": llm_answer,
                "gold": correct_answer_normalized,
                "correct": is_correct,
                "topics": topics
            })
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Print per-topic accuracy
    print("\nPer-Topic Accuracy:")
    topic_results = {}
    for topic in sorted(total_by_topic.keys()):
        topic_acc = correct_by_topic[topic] / total_by_topic[topic]
        topic_results[topic] = {
            "accuracy": topic_acc,
            "correct": correct_by_topic[topic],
            "total": total_by_topic[topic]
        }
        print(f"  {topic}: {topic_acc:.4f} ({correct_by_topic[topic]}/{total_by_topic[topic]})")
    
    # Save detailed results
    if detailed_output is not None:
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"Detailed results saved to: {detailed_output}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_topic": topic_results
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate on CTI-Bench and Additional Benchmarks")
    
    # Model loading options
    parser.add_argument("--model_path", type=str, help="Path to model (base or fine-tuned) - for local inference")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name (required for LoRA)")
    parser.add_argument("--is_base", action="store_true", help="Evaluate base model (pre-training)")
    
    # API endpoint options
    parser.add_argument("--use_api", action="store_true", help="Use API endpoint instead of local model")
    parser.add_argument("--api_endpoint", type=str, help="OpenAI-compatible API endpoint (e.g., http://IP:7799/v1/chat/completions)")
    parser.add_argument("--api_model", type=str, help="Model name for API endpoint")
    parser.add_argument("--api_key", type=str, default="", help="API key if needed (leave empty for local vLLM)")
    
    # Evaluation options
    parser.add_argument("--tasks", nargs="+", default=["mcq", "rcm", "vsp", "ate"], 
                       help="Tasks to evaluate (mcq, rcm, vsp, ate, cybermetric, seceval, cissp)")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples per task (for testing)")
    
    # Dataset paths for external benchmarks
    parser.add_argument("--cissp_path", type=str, default=None, help="Path to CISSP dataset JSON file")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_api:
        if not args.api_endpoint or not args.api_model:
            parser.error("--api_endpoint and --api_model are required when --use_api is set")
        model = None
        tokenizer = None
    else:
        if not args.model_path:
            parser.error("--model_path is required for local inference (or use --use_api)")
        # Load model for local inference
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model, args.is_base)
    
    # Generate output filename from model path if not specified
    if args.output is None:
        if args.use_api:
            model_name = args.api_model.rstrip('/').split('/')[-1]
        else:
            model_name = args.model_path.rstrip('/').split('/')[-1]
        # Sanitize filename
        model_name = model_name.replace('/', '-').replace('\\', '-')
        args.output = f"eval_results_{model_name}.json"
    
    # Create detailed output directory
    detailed_dir = args.output.replace('.json', '_detailed')
    os.makedirs(detailed_dir, exist_ok=True)
    print(f"Detailed results will be saved to: {detailed_dir}/")
    
    # Prepare API kwargs for generate_response
    api_kwargs = {
        'use_api': args.use_api,
        'api_endpoint': args.api_endpoint,
        'api_model': args.api_model,
        'api_key': args.api_key
    }
    
    # Run evaluations
    results = {
        "model_path": args.model_path if not args.use_api else args.api_model,
        "evaluation_mode": "api" if args.use_api else "local",
        "is_base_model": args.is_base,
        "tasks": {}
    }
    
    task_map = {
        "mcq": ("AI4Sec/cti-bench", "cti-mcq", evaluate_mcq),
        "rcm": ("AI4Sec/cti-bench", "cti-rcm", evaluate_rcm),
        "vsp": ("AI4Sec/cti-bench", "cti-vsp", evaluate_vsp),
        "ate": ("AI4Sec/cti-bench", "cti-ate", evaluate_ate),
        "cybermetric": (None, None, evaluate_cybermetric),
        "seceval": (None, None, evaluate_seceval),
        "cissp": (None, None, evaluate_cissp),
    }
    
    for task_name in args.tasks:
        if task_name not in task_map:
            print(f"Unknown task: {task_name}, skipping...")
            continue
        
        dataset_name, subset_name, eval_fn = task_map[task_name]
        detailed_output = os.path.join(detailed_dir, f"{task_name}_detailed.jsonl")
        
        # Special handling for external JSON benchmarks (CyberMetric, SecEval, CISSP)
        if task_name in ["cybermetric", "seceval", "cissp"]:
            # Determine dataset path
            if task_name == "cissp":
                dataset_path = args.cissp_path
            else:
                dataset_path = None  # Uses default URLs
            
            task_results = eval_fn(model, tokenizer, dataset_path=dataset_path, 
                                 detailed_output=detailed_output, **api_kwargs)
            results["tasks"][task_name.upper()] = task_results
        else:
            print(f"\nLoading {subset_name} dataset...")
            dataset = load_dataset(dataset_name, subset_name, split="test")
            
            if args.max_samples:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            
            # Run evaluation with detailed output
            task_results = eval_fn(model, tokenizer, dataset, detailed_output=detailed_output, **api_kwargs)
            results["tasks"][task_name.upper()] = task_results
    
    # Save results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(json.dumps(results, indent=2))
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    print(f"Detailed results directory: {detailed_dir}/")


if __name__ == "__main__":
    main()
