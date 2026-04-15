#!/usr/bin/env python3
"""
Evaluate fine-tuned model using LLM-as-a-Judge approach

Instead of regex-based extraction, uses an LLM judge to determine if answers are correct.
This is more robust for handling various answer formats and reasoning chains.

Supports all benchmarks:
- CTI-Bench (RISys-Lab): MCQ, RCM, VSP, ATE
- AthenaBench (GitHub): CKT, RMS, TAA
- Other: CyberMetric, SecEval, CISSP, MMLU-CS, SECURE, SecBench
"""

import os
import re
import json
import torch
import requests
import argparse
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional vLLM import for fast batched judge inference
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


# Import core functions from evaluate.py
from evaluate import (
    load_model_and_tokenizer,
    chat_completion_api,
    generate_response,
    parse_ids_from_text
)


def load_jsonl_dataset(source: str) -> HFDataset:
    """Load JSONL dataset from GitHub URL or local file path
    
    Args:
        source: Either a GitHub raw URL or local file path to .jsonl file
    
    Returns:
        HuggingFace Dataset object
    """
    data = []
    
    # Check if source is a URL or local path
    if source.startswith('http://') or source.startswith('https://'):
        print(f"  Downloading from: {source}")
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
    else:
        print(f"  Loading from local file: {source}")
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
    
    print(f"  Loaded {len(data)} samples")
    return HFDataset.from_list(data)


def create_judge_prompt(task_type: str, question: str, model_answer: str, 
                       ground_truth: str, extra_context: dict = None) -> str:
    """Create a prompt for the LLM judge to evaluate model answer
    
    This function acts as a TEMPLATE SELECTOR - it returns a different evaluation
    prompt based on the task type. Each template is optimized for that task's
    specific requirements.
    
    STRUCTURE FOR EACH TASK:
    ┌─────────────────────────────────────┐
    │ 1. Context (what are you judging?)  │
    │ 2. Question                          │
    │ 3. Correct Answer                    │
    │ 4. Model's Response                  │
    │ 5. Evaluation Instructions           │
    │ 6. Output Format (CORRECT/INCORRECT) │
    └─────────────────────────────────────┘
    
    TASK TYPES:
    - mcq/cybermetric/cissp/mmlu_cs/secure/secbench/ckt: Single/multi-choice questions (A/B/C/D/E)
    - seceval: Multi-select questions (can be ABC, AD, etc.)
    - rcm: CWE ID extraction (CWE-79, etc.)
    - vsp: CVSS score prediction (0.0-10.0) - regression task, returns MAD
    - ate: MITRE ATT&CK techniques (T1234, T1234.567)
    - rms: Risk mitigation strategies (M1018, M1026, etc.)
    - taa: Threat actor attribution (actor names)
    
    Args:
        task_type: Type of task (mcq, rcm, vsp, ate, cybermetric, seceval, cissp, mmlu_cs, secure, secbench, ckt, rms, taa)
        question: Original question/prompt
        model_answer: Model's response
        ground_truth: Correct answer
        extra_context: Additional context (e.g., answer choices for MCQ)
    
    Returns:
        Formatted prompt for judge
    """
    
    # ============================================================
    # SINGLE-CHOICE MCQ: mcq, cybermetric, cissp, mmlu_cs, secure, secbench, ckt
    # Expected answer: A single letter (A, B, C, D, or E for CKT)
    # ============================================================
    if task_type in ["mcq", "cybermetric", "cissp", "mmlu_cs", "secure", "secbench", "ckt"]:
        choices_info = ""
        if extra_context and "choices" in extra_context:
            choices_info = f"\n\nAnswer Choices:\n{extra_context['choices']}"
        
        return f"""You are evaluating a model's answer to a multiple-choice cybersecurity question.

Question:
{question}{choices_info}

Correct Answer: {ground_truth}

Model's Response:
{model_answer}

Task: Determine if the model's response contains the correct answer ({ground_truth}).
The model may provide reasoning before the answer. Look for the final answer in the response.

Respond ONLY with a JSON object in this exact format:
{{
  "verdict": "CORRECT" or "INCORRECT",
  "justification": "one-sentence explanation of your decision"
}}

Your judgment:"""

    # ============================================================
    # MULTI-SELECT MCQ: seceval
    # Expected answer: Multiple letters (e.g., "ABC", "AD", "B")
    # Logic from evaluate.py: Extract A-D, deduplicate, sort, then compare
    # "CAB" and "ABC" are both normalized to "ABC" → match
    # "AB" vs "ABC" → no match (missing C)
    # "ABCD" vs "ABC" → no match (extra D)
    # ============================================================
    elif task_type == "seceval":
        return f"""You are evaluating a model's answer to a multi-select cybersecurity question.

Question:
{question}

Correct Answer: {ground_truth}

Model's Response:
{model_answer}

Task: Determine if the model selected the exact same SET of options as the correct answer.
Extract all letters A-D from the model's response, deduplicate and sort them, then compare.
Examples:
- Model says "CAB" or "ABC" or "A, B, C" → all become "ABC"
- If correct is "ABC": "ABC" is CORRECT, "AB" is INCORRECT (missing), "ABCD" is INCORRECT (extra)

Respond ONLY with a JSON object in this exact format:
{{
  "verdict": "CORRECT" or "INCORRECT",
  "justification": "one-sentence explanation of your decision"
}}

Your judgment:"""

    # ============================================================
    # CWE IDENTIFICATION: rcm (Root Cause Mapping)
    # Expected answer: CWE IDs (e.g., "CWE-79", "CWE-89")
    # May have multiple correct CWEs
    # ============================================================
    elif task_type == "rcm":
        return f"""You are evaluating a model's CWE identification for a CVE.

Question:
{question}

Correct CWE ID(s): {ground_truth}

Model's Response:
{model_answer}

Task: Determine if the model identified the correct CWE ID(s).
Look for CWE identifiers in format "CWE-XXX". Multiple CWEs may be correct.

Respond ONLY with a JSON object in this exact format:
{{
  "verdict": "CORRECT" or "INCORRECT",
  "justification": "one-sentence explanation of your decision"
}}

Your judgment:"""

    # ============================================================
    # CVSS VECTOR: vsp (Vulnerability Severity Prediction)
    # Expected answer: CVSS vector string (e.g., CVSS:3.1/AV:N/AC:L/...)
    # Judge extracts vector, then Python calculates MAD using CVSS library
    # ============================================================
    elif task_type == "vsp":
        return f"""You are extracting CVSS vectors from security assessment responses.

Question:
{question}

Model's Response:
{model_answer}

Task: Extract the CVSS v3.1 vector from the model's response.

CRITICAL INSTRUCTIONS:
1. Find the CVSS vector string in the response (format: CVSS:3.1/AV:X/AC:X/PR:X/UI:X/S:X/C:X/I:X/A:X or AV:X/AC:X/...)
2. Normalize the prefix:
   - If it starts with CVSS:3.0/, change to CVSS:3.1/
   - If it starts with CVSS:3.1/, keep as is
   - If no prefix, add CVSS:3.1/
3. If no valid CVSS vector is found in the response, set extraction_success to false
4. Only extract the vector components (AV, AC, PR, UI, S, C, I, A) - ignore any numerical scores

Respond ONLY with a JSON object in this exact format:
{{
  "extracted_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
  "extraction_success": true
}}

Your extraction:"""

    # ============================================================
    # MITRE TECHNIQUES: ate (ATT&CK Technique Extraction)
    # Expected answer: Technique IDs (e.g., "T1234", "T1234.567")
    # Special rule: T1234.567 and T1234 may be equivalent
    # ============================================================
    elif task_type == "ate":
        return f"""You are evaluating a model's MITRE ATT&CK technique identification.

Question:
{question}

Correct Technique ID(s): {ground_truth}

Model's Response:
{model_answer}

Task: Determine if the model identified the correct MITRE ATT&CK technique(s).
Look for technique IDs in format "T####" or "T####.###" (subtechniques).
Note: T1234.567 and T1234 may be considered equivalent (subtechnique vs parent).

Respond ONLY with a JSON object in this exact format:
{{
  "verdict": "CORRECT" or "INCORRECT",
  "justification": "one-sentence explanation of your decision"
}}

Your judgment:"""

    # ============================================================
    # RMS: Risk Mitigation Strategies
    # Expected answer: Comma-separated MITRE ATT&CK mitigation IDs (M1018, M1026, M1028, M1047)
    # ============================================================
    elif task_type == "rms":
        return f"""You are evaluating a model's answer to a risk mitigation strategy question.

Question:
{question}

Correct Answer: {ground_truth}

Model's Response:
{model_answer}

Task: Determine if the model identified the correct MITRE ATT&CK mitigation IDs.
Look for mitigation IDs in format "M10xx" (e.g., M1037, M1041).
The model should identify the same set of mitigations, though order doesn't matter.

Respond ONLY with a JSON object in this exact format:
{{
  "verdict": "CORRECT" or "INCORRECT",
  "justification": "one-sentence explanation of your decision"
}}

Your judgment:"""

    # ============================================================
    # TAA: Threat Actor Attribution
    # Expected answer: Threat actor name (e.g., "TraderTraitor", "Lazarus")
    # ============================================================
    elif task_type == "taa":
        return f"""You are evaluating a model's answer to a threat actor attribution question.

Question:
{question}

Correct Answer: {ground_truth}

Model's Response:
{model_answer}

Task: Determine if the model identified the correct threat actor.
The model's answer should match the expected threat actor name.
Account for minor variations in naming (e.g., "Lazarus Group" vs "Lazarus").

Respond ONLY with a JSON object in this exact format:
{{
  "verdict": "CORRECT" or "INCORRECT",
  "justification": "one-sentence explanation of your decision"
}}

Your judgment:"""

    # ============================================================
    # SEvenLLM: Structured JSON Extraction Tasks
    # Uses SEvenLLM's GPT-4 scoring approach: 1-5 scale
    # Reference: https://github.com/CSJianYang/SEevenLLM/blob/main/code/score/get_score_gpt4_prompt.py
    # ============================================================
    elif task_type == "sevenllm":
        return f"""As a knowledge analysis expert in the field of cybersecurity, please rate the following network security event Q&A results.

The content of the network security incident is:
{extra_context.get('input', question) if extra_context else question}

The question raised regarding the content of this cybersecurity incident is:
{question}

The expected answer (ground truth) is:
{ground_truth}

The model's answer provided for this event's content and question is:
{model_answer}

Scoring method:
Using a 5-point scoring system:
- 1 point: very poor (completely wrong or irrelevant)
- 2 points: slightly poor (major errors or missing key information)
- 3 points: usable (partially correct but incomplete)
- 4 points: good (mostly correct with minor issues)
- 5 points: very good (accurate, complete, matches expected output)

Scoring reference criteria:
- Whether the answer is answered accurately as required by the question
- Whether the format is regular and the language is coherent and readable
- Whether the extracted information matches the expected output
- Whether the overall content has information value in the field of network security

Respond ONLY with a JSON object in this exact format:
{{
  "score": <1-5>,
  "reason": "explanation of your score"
}}

Your evaluation:"""

    else:
        # mmlu_cs is handled by the mcq branch above (task_type in ["mcq", ..., "mmlu_cs", ...])
        raise ValueError(f"Unknown task_type: {task_type}. Supported: mcq, rcm, vsp, ate, cybermetric, seceval, cissp, mmlu_cs, secure, secbench, ckt, rms, taa, sevenllm")


def parse_judge_response(judge_response: str, task_type: str) -> Dict[str, Any]:
    """Parse judge response to extract verdict or extraction result
    
    This is the shared parsing logic used by both evaluate_llm_judge and test_llm_judge.
    
    Args:
        judge_response: Raw response from judge LLM
        task_type: Type of task (mcq, rcm, vsp, ate, etc.)
    
    Returns:
        For VSP:
            dict with keys:
                - extracted_vector: str (CVSS vector)
                - extraction_success: bool
                - judge_response: str (raw judge output)
        For other tasks:
            dict with keys:
                - is_correct: bool
                - judge_response: str (raw judge output)
                - justification: str (judge's explanation)
    """
    # Parse JSON response with error handling
    try:
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', judge_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'{.*}', judge_response, re.DOTALL)
            json_str = json_match.group(0) if json_match else judge_response
        
        parsed = json.loads(json_str)
        
        # VSP uses extraction format (different from other tasks)
        if task_type == "vsp":
            extracted_vector = parsed.get("extracted_vector", "")
            extraction_success = parsed.get("extraction_success", False)
            
            return {
                "extracted_vector": extracted_vector,
                "extraction_success": extraction_success,
                "judge_response": judge_response
            }
        
        # For other tasks, extract verdict
        verdict = parsed.get("verdict", "").strip().upper()
        justification = parsed.get("justification", "No justification provided")
        
        # Handle sevenllm with SEvenLLM's 1-5 scoring scale
        if task_type == "sevenllm":
            score = parsed.get("score", 1)
            # Normalize to 1-5 if somehow got float
            if isinstance(score, float) and score <= 1.0:
                score = int(score * 5)
            score = max(1, min(5, int(score)))  # Clamp to 1-5
            reason = parsed.get("reason", "No reason provided")
            # Consider score >= 4 as "correct" for accuracy metrics
            is_correct = score >= 4
            return {
                "is_correct": is_correct,
                "score": score,
                "reason": reason,
                "judge_response": judge_response,
                "justification": reason
            }
        
        is_correct = verdict == "CORRECT"
        
        return {
            "is_correct": is_correct,
            "judge_response": judge_response,
            "justification": justification
        }
        
    except (json.JSONDecodeError, AttributeError) as e:
        # Fallback to legacy parsing if JSON parsing fails
        print(f"Warning: Failed to parse JSON from judge response: {e}")
        print(f"Raw response: {judge_response[:200]}...")
        
        if task_type == "vsp":
            # For VSP, try to extract CVSS vector with regex as fallback
            cvss_pattern = r'(?:CVSS:3\.[01]/)?AV:[A-Z]+/AC:[A-Z]+/PR:[A-Z]+/UI:[A-Z]+/S:[A-Z]+/C:[A-Z]+/I:[A-Z]+/A:[A-Z]+'
            match = re.search(cvss_pattern, judge_response, re.IGNORECASE)
            if match:
                return {
                    "extracted_vector": match.group(0),
                    "extraction_success": True,
                    "judge_response": judge_response
                }
            return {
                "extracted_vector": "",
                "extraction_success": False,
                "judge_response": judge_response
            }
        
        # For sevenllm, fallback to extracting score from text
        if task_type == "sevenllm":
            # Try to find a number 1-5 in the response
            score_match = re.search(r'["\']?score["\']?\s*[:=]\s*(\d)', judge_response)
            if score_match:
                score = int(score_match.group(1))
            else:
                # Fallback: look for any digit 1-5
                digits = re.findall(r'\b([1-5])\b', judge_response)
                score = int(digits[0]) if digits else 3  # Default to 3 (usable)
            score = max(1, min(5, score))
            is_correct = score >= 4
            return {
                "is_correct": is_correct,
                "score": score,
                "reason": "Fallback parsing - JSON parse failed",
                "judge_response": judge_response,
                "justification": "Fallback parsing - JSON parse failed"
            }
        
        # For other tasks, fallback to text matching
        judge_response_upper = judge_response.strip().upper()
        is_correct = "CORRECT" in judge_response_upper and "INCORRECT" not in judge_response_upper
        
        return {
            "is_correct": is_correct,
            "judge_response": judge_response,
            "justification": "Fallback parsing - JSON parse failed"
        }


def calculate_vsp_mad(pred_vector: str, gold_vector: str) -> float:
    """Calculate Mean Absolute Deviation (MAD) between CVSS vectors
    
    Uses the RedSage/Athena approach:
    1. Parse both vectors using CVSS3 library
    2. Extract numerical base scores
    3. Return absolute difference
    
    Args:
        pred_vector: Predicted CVSS vector (with or without prefix)
        gold_vector: Ground truth CVSS vector (with or without prefix)
    
    Returns:
        MAD score (0.0 = perfect match, 10.0 = maximum difference/error)
    """
    try:
        from cvss import CVSS3
        
        # Normalize prefixes (add CVSS:3.1/ if missing, convert 3.0 to 3.1)
        def normalize_vector(v: str) -> str:
            v = v.strip()
            if v.startswith('CVSS:3.0/'):
                return v.replace('CVSS:3.0/', 'CVSS:3.1/')
            elif v.startswith('CVSS:3.1/'):
                return v
            else:
                # No prefix, add CVSS:3.1/
                return 'CVSS:3.1/' + v
        
        pred_normalized = normalize_vector(pred_vector)
        gold_normalized = normalize_vector(gold_vector)
        
        # Parse vectors and extract base scores
        pred_cvss = CVSS3(pred_normalized)
        gold_cvss = CVSS3(gold_normalized)
        
        pred_score = pred_cvss.scores()[0]  # Base score (0.0-10.0)
        gold_score = gold_cvss.scores()[0]
        
        # Return absolute difference
        mad = abs(pred_score - gold_score)
        return round(mad, 2)
        
    except Exception as e:
        # Return max penalty (10.0) for invalid vectors
        print(f"Warning: Failed to calculate CVSS MAD: {e}")
        return 10.0


def initialize_judge_vllm(model_path: str, gpu_memory_utilization: float = 0.9) -> "LLM":
    """Initialize vLLM LLM for fast batch judge inference
    
    Args:
        model_path: Path to judge model
        gpu_memory_utilization: GPU memory fraction to use
        
    Returns:
        vLLM LLM object
    """
    if not HAS_VLLM:
        raise RuntimeError(
            "vLLM not installed. Install with: pip install vllm\n"
            "Or use --judge_use_api for API-based judge."
        )
    
    print(f"Initializing vLLM judge with model: {model_path}")
    print(f"GPU memory utilization: {gpu_memory_utilization*100:.0f}%")
    
    judge_vllm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        disable_log_stats=True,
        enable_prefix_caching=False
    )
    
    return judge_vllm


def generate_judge_responses_vllm(judge_vllm: "LLM", prompts: list, 
                                 max_tokens: int = 256,
                                 temperature: float = 0.0) -> list:
    """Generate judge responses using vLLM batch inference
    
    Args:
        judge_vllm: vLLM LLM object
        prompts: List of judge prompts to evaluate
        max_tokens: Maximum tokens per response
        temperature: Sampling temperature (0.0 for deterministic)
        
    Returns:
        List of judge responses
    """
    # Get tokenizer from vLLM to apply chat templates
    tokenizer = judge_vllm.get_tokenizer()
    
    # Apply chat templates to all prompts (same as non-vLLM version)
    formatted_prompts = []
    for prompt in prompts:
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback if chat template not available
            formatted_prompt = prompt
        formatted_prompts.append(formatted_prompt)
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
    )
    
    # Generate responses in batch
    outputs = judge_vllm.generate(formatted_prompts, sampling_params)
    
    # Extract text from outputs
    responses = [output.outputs[0].text.strip() for output in outputs]
    return responses


def judge_answer(judge_model, judge_tokenizer, task_type: str, question: str,
                model_answer: str, ground_truth: str, 
                extra_context: dict = None, judge_vllm = None, **api_kwargs) -> dict:
    """Use LLM judge to evaluate if model answer is correct
    
    For VSP task: Judge extracts CVSS vector, then MAD is calculated separately
    For other tasks: Judge provides binary judgment (CORRECT/INCORRECT)
    
    Returns:
        dict with keys:
            - is_correct: bool
            - judge_response: str (raw judge output)
            - justification: str (judge's explanation)
    """
    
    # Create judge prompt
    judge_prompt = create_judge_prompt(
        task_type, question, model_answer, ground_truth, extra_context
    )
    
    # Get judge's response
    if judge_vllm:
        # Use vLLM for single judge inference (for compatibility)
        judge_response_list = generate_judge_responses_vllm(judge_vllm, [judge_prompt], max_tokens=256)
        judge_response = judge_response_list[0]
    else:
        # Use standard generation (HuggingFace transformers or API)
        judge_response = generate_response(
            judge_model, judge_tokenizer, judge_prompt,
            max_new_tokens=256,  # Shorter for judge
            **api_kwargs
        )
    
    # Parse response using shared parsing logic
    return parse_judge_response(judge_response, task_type)


def evaluate_with_judge(model, tokenizer, dataset, task_type: str,
                       judge_model, judge_tokenizer, judge_vllm=None, detailed_output=None,
                       **api_kwargs):
    """Generic evaluation function using LLM judge
    
    All tasks use binary verdict (CORRECT/INCORRECT):
    - Classification tasks (MCQ, RCM, ATE, SecEval, CyberMetric, CISSP, MMLU-CS, SECURE, SecBench): Direct answer matching
    - VSP: CVSS vector component-by-component comparison
    
    Returns accuracy, correct count, and total for all task types.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for model
        dataset: Dataset to evaluate on
        task_type: Type of task (mcq, rcm, vsp, ate, etc.)
        judge_model: Judge model
        judge_tokenizer: Judge tokenizer
        detailed_output: Path to save detailed results
        **api_kwargs: API kwargs for both model and judge
    """
    
    correct = 0
    total = 0
    detailed_results = []
    mad_scores = []  # For VSP MAD calculation
    extraction_success_count = 0  # For VSP extraction tracking
    
    # For ATE metrics (P/R/F1)
    tp_total = 0
    fp_total = 0
    fn_total = 0
    exact_matches = 0
    
    for idx, sample in enumerate(tqdm(dataset, desc=f"Evaluating {task_type.upper()}")):
        # Normalize field names to standard format
        # Priority: Prompt (CTI-Bench) > prompt (SECURE) > question (others)
        question = sample.get('Prompt') or sample.get('prompt') or sample.get('question')
        if not question:
            continue
        
        # Normalize answer field: GT > solution > answer > correct_answer > label
        ground_truth = sample.get('GT') or sample.get('solution')
        if not ground_truth:
            answer_val = sample.get('answer') or sample.get('correct_answer') or sample.get('label')
            if answer_val:
                # Handle MMLU-CS format: answer is integer index
                if isinstance(answer_val, int):
                    ground_truth = ['A', 'B', 'C', 'D'][answer_val]
                else:
                    ground_truth = str(answer_val).strip()
            else:
                ground_truth = ""
        else:
            ground_truth = str(ground_truth).strip()
        
        # Get model's response
        response = generate_response(model, tokenizer, question, 
                                    max_new_tokens=1024, **api_kwargs)
        
        # Prepare extra context
        extra_context = {}
        if 'answers' in sample:  # CyberMetric or SecBench format
            extra_context['choices'] = sample['answers']
        elif 'choices' in sample:  # SecEval or MMLU-CS format
            extra_context['choices'] = sample['choices']
        elif 'options' in sample:  # SECURE format
            extra_context['choices'] = sample['options']
        elif 'option_a' in sample:  # AthenaBench JSONL format (CKT, RMS, TAA)
            # Format 5-option choices for CKT or other AthenaBench tasks
            choices_list = []
            for opt_key in ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']:
                if opt_key in sample:
                    letter = opt_key[-1].upper()
                    choices_list.append(f"{letter}) {sample[opt_key]}")
            extra_context['choices'] = '\n'.join(choices_list)
        
        # Get judge's evaluation (do NOT forward model api_kwargs to the judge)
        judge_result = judge_answer(
            judge_model, judge_tokenizer, task_type,
            question, response, ground_truth, extra_context,
            judge_vllm=judge_vllm,
        )
        
        # Handle VSP (regression) vs other tasks (classification)
        if task_type == "vsp":
            # For VSP: extract vector and calculate MAD
            extracted_vector = judge_result.get('extracted_vector', '')
            extraction_success = judge_result.get('extraction_success', False)
            
            if extraction_success:
                extraction_success_count += 1
                # Calculate MAD between extracted and ground truth vectors
                mad = calculate_vsp_mad(extracted_vector, ground_truth)
                mad_scores.append(mad)
            else:
                # Failed extraction gets max penalty
                mad = 10.0
                mad_scores.append(mad)
            
            total += 1
            
            # Save detailed result for VSP
            if detailed_output is not None:
                result = {
                    "index": idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_response": response,
                    "extracted_vector": extracted_vector,
                    "judge_response": judge_result['judge_response'],
                    "mad": mad,
                    "extraction_success": extraction_success
                }
                detailed_results.append(result)
        else:
            # For classification tasks: count correct/incorrect
            is_correct = judge_result['is_correct']
            
            if is_correct:
                correct += 1
            
            total += 1
            
            # For ATE: extract techniques and calculate precision/recall/F1
            if task_type == "ate":
                # parse_judge_response does not return extracted_answer; fall back to raw response
                judge_answer_text = judge_result.get('extracted_answer') or response
                pred_techniques = parse_ids_from_text(judge_answer_text)
                gold_techniques = parse_ids_from_text(ground_truth)
                
                tp = len(pred_techniques & gold_techniques)
                fp = len(pred_techniques - gold_techniques)
                fn = len(gold_techniques - pred_techniques)
                
                tp_total += tp
                fp_total += fp
                fn_total += fn
                
                if pred_techniques == gold_techniques:
                    exact_matches += 1
                
                if detailed_output is not None:
                    result = {
                        "index": idx,
                        "question": question,
                        "ground_truth": ground_truth,
                        "model_response": response,
                        "pred_techniques": sorted(list(pred_techniques)),
                        "gold_techniques": sorted(list(gold_techniques)),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "exact_match": pred_techniques == gold_techniques
                    }
                    detailed_results.append(result)
            else:
                if detailed_output is not None:
                    result = {
                        "index": idx,
                        "question": question,
                        "ground_truth": ground_truth,
                        "model_response": response,
                        "judge_response": judge_result['judge_response'],
                        "judge_justification": judge_result.get('justification', ''),
                        "is_correct": is_correct
                    }
                    # Add score for sevenllm (1-5 scale)
                    if 'score' in judge_result:
                        result['score'] = judge_result['score']
                    detailed_results.append(result)
    
    # Calculate metrics based on task type
    if task_type == "vsp":
        # VSP uses MAD metric (lower is better)
        mean_mad = sum(mad_scores) / len(mad_scores) if mad_scores else 10.0
        # Normalize to accuracy-like metric using denominator 7.7 (Athena standard)
        vsp_accuracy = max(0.0, 1.0 - (mean_mad / 7.7))
        
        results = {
            "mad": round(mean_mad, 3),
            "accuracy": round(vsp_accuracy, 3),
            "total": total,
            "extraction_success_count": extraction_success_count
        }
    elif task_type == "ate":
        # ATE uses micro-averaged precision/recall/F1
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        exact_match_rate = exact_matches / total if total > 0 else 0.0
        
        print(f"\nMicro-averaged metrics (ATE):")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Exact Match: {exact_match_rate:.4f}")
        print(f"Total TP/FP/FN: {tp_total}/{fp_total}/{fn_total}")
        
        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": exact_match_rate,
            "tp_total": tp_total,
            "fp_total": fp_total,
            "fn_total": fn_total
        }
    else:
        # Classification tasks use accuracy
        accuracy = correct / total if total > 0 else 0.0
        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    # Save detailed results
    if detailed_output is not None:
        os.makedirs(os.path.dirname(detailed_output), exist_ok=True)
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"Detailed results saved to: {detailed_output}")
    
    return results


def evaluate_with_judge_from_responses(dataset, task_type: str,
                                       judge_model, judge_tokenizer, judge_vllm=None,
                                       detailed_output: str = None,
                                       **judge_api_kwargs) -> Dict[str, Any]:
    """Evaluate model using LLM judge on pre-collected responses (NO INFERENCE)
    
    This function evaluates responses that were already collected by run_inference_benchmarks.py.
    It only runs the judge, not the model being evaluated.
    
    Args:
        dataset: Dataset with 'Prompt', 'GT', 'model_response' fields
        task_type: Type of task
        judge_model: Judge model
        judge_tokenizer: Judge tokenizer  
        detailed_output: Path to save detailed results
        **judge_api_kwargs: API kwargs for judge
    
    Returns:
        Results dict with accuracy/MAD metrics
    """
    correct = 0
    total = 0
    detailed_results = []
    mad_scores = []
    extraction_success_count = 0
    
    # For ATE metrics (P/R/F1)
    tp_total = 0
    fp_total = 0
    fn_total = 0
    exact_matches = 0
    
    for idx, sample in enumerate(tqdm(dataset, desc=f"Judging {task_type.upper()}")):
        # Handle both pre-collected format (prompt/ground_truth) and legacy format (Prompt/GT)
        question = sample.get('prompt', sample.get('Prompt', ''))
        ground_truth = sample.get('ground_truth', sample.get('GT', ''))
        response = sample['model_response']
        
        # Prepare extra context from metadata
        extra_context = {}
        metadata = sample.get('metadata', {})
        if 'choices' in metadata:
            choices = metadata['choices']
            if isinstance(choices, dict):
                extra_context['choices'] = '\n'.join([f"{k}. {v}" for k, v in sorted(choices.items())])
            elif isinstance(choices, list):
                extra_context['choices'] = choices
        
        # For SEvenLLM: pass input (cybersecurity incident content) for full context
        if 'input' in metadata:
            extra_context['input'] = metadata['input']
        
        # Get judge's evaluation
        judge_result = judge_answer(
            judge_model, judge_tokenizer, task_type,
            question, response, ground_truth, extra_context,
            judge_vllm=judge_vllm,
            **judge_api_kwargs
        )
        
        # Handle VSP (regression) vs other tasks (classification)
        if task_type == "vsp":
            extracted_vector = judge_result.get('extracted_vector', '')
            extraction_success = judge_result.get('extraction_success', False)
            
            if extraction_success:
                extraction_success_count += 1
                mad = calculate_vsp_mad(extracted_vector, ground_truth)
                mad_scores.append(mad)
            else:
                mad = 10.0
                mad_scores.append(mad)
            
            total += 1
            
            if detailed_output is not None:
                result = {
                    "index": idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_response": response,
                    "extracted_vector": extracted_vector,
                    "judge_response": judge_result['judge_response'],
                    "mad": mad,
                    "extraction_success": extraction_success
                }
                detailed_results.append(result)
        else:
            is_correct = judge_result['is_correct']
            
            if is_correct:
                correct += 1
            
            total += 1
            
            # For ATE: extract techniques and calculate precision/recall/F1
            if task_type == "ate":
                # parse_judge_response does not return extracted_answer; fall back to raw response
                judge_answer_text = judge_result.get('extracted_answer') or response
                pred_techniques = parse_ids_from_text(judge_answer_text)
                gold_techniques = parse_ids_from_text(ground_truth)
                
                tp = len(pred_techniques & gold_techniques)
                fp = len(pred_techniques - gold_techniques)
                fn = len(gold_techniques - pred_techniques)
                
                tp_total += tp
                fp_total += fp
                fn_total += fn
                
                if pred_techniques == gold_techniques:
                    exact_matches += 1
                
                if detailed_output is not None:
                    result = {
                        "index": idx,
                        "question": question,
                        "ground_truth": ground_truth,
                        "model_response": response,
                        "pred_techniques": sorted(list(pred_techniques)),
                        "gold_techniques": sorted(list(gold_techniques)),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "exact_match": pred_techniques == gold_techniques
                    }
                    detailed_results.append(result)
            else:
                if detailed_output is not None:
                    result = {
                        "index": idx,
                        "question": question,
                        "ground_truth": ground_truth,
                        "model_response": response,
                        "judge_response": judge_result['judge_response'],
                        "judge_justification": judge_result.get('justification', ''),
                        "is_correct": is_correct
                    }
                    # Add score for sevenllm (1-5 scale)
                    if 'score' in judge_result:
                        result['score'] = judge_result['score']
                    detailed_results.append(result)
    
    # Calculate metrics
    if task_type == "vsp":
        mean_mad = sum(mad_scores) / len(mad_scores) if mad_scores else 10.0
        vsp_accuracy = max(0.0, 1.0 - (mean_mad / 7.7))
        
        results = {
            "mad": round(mean_mad, 3),
            "accuracy": round(vsp_accuracy, 3),
            "total": total,
            "extraction_success_count": extraction_success_count
        }
    elif task_type == "ate":
        # ATE uses micro-averaged precision/recall/F1
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        exact_match_rate = exact_matches / total if total > 0 else 0.0
        
        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": exact_match_rate,
            "tp_total": tp_total,
            "fp_total": fp_total,
            "fn_total": fn_total
        }
    else:
        accuracy = correct / total if total > 0 else 0.0
        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    # Save detailed results
    if detailed_output is not None:
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
    
    return results


def parse_concatenated_json(content: str) -> List[Dict]:
    """Parse JSON objects directly concatenated without separators.
    
    Handles JSONL files where records may be concatenated with escaped newlines.
    """
    objects = []
    i = 0
    
    while i < len(content):
        # Skip whitespace and non-JSON chars
        while i < len(content) and content[i] not in '{[':
            i += 1
        
        if i >= len(content):
            break
        
        # Find the matching closing brace/bracket
        if content[i] == '{':
            brace_count = 0
            start = i
            while i < len(content):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            obj_str = content[start:i+1]
                            obj = json.loads(obj_str)
                            objects.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i += 1
                        break
                i += 1
        else:
            i += 1
    
    return objects


def generate_summary_from_detailed(detailed_dir: str) -> Dict[str, Any]:
    """Generate summary from detailed JSONL results.
    
    Args:
        detailed_dir: Directory containing *_detailed.jsonl files
        
    Returns:
        Dictionary with summary metrics for each task
    """
    summary = {
        "tasks": {},
        "total_correct": 0,
        "total_samples": 0
    }
    
    # Process each task file
    detailed_path = Path(detailed_dir)
    for f in sorted(detailed_path.glob('*_detailed.jsonl')):
        task = f.stem.replace('_detailed', '')
        content = f.read_text()
        objects = parse_concatenated_json(content)
        
        if not objects:
            continue
        
        # Calculate metrics
        total = len(objects)
        correct = sum(1 for obj in objects if obj.get('is_correct'))
        
        task_metrics = {
            'accuracy': correct / total if total > 0 else 0,
            'correct': correct,
            'total': total
        }
        
        # Add precision/recall/f1 if present
        if objects and 'precision' in objects[0]:
            precisions = [obj.get('precision', 0) for obj in objects]
            recalls = [obj.get('recall', 0) for obj in objects]
            f1_scores = [obj.get('f1', 0) for obj in objects]
            
            task_metrics['avg_precision'] = sum(precisions) / len(precisions) if precisions else 0
            task_metrics['avg_recall'] = sum(recalls) / len(recalls) if recalls else 0
            task_metrics['avg_f1'] = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        
        summary['tasks'][task.upper()] = task_metrics
        summary['total_correct'] += correct
        summary['total_samples'] += total
    
    # Calculate overall accuracy
    summary['overall_accuracy'] = (summary['total_correct'] / summary['total_samples'] 
                                   if summary['total_samples'] > 0 else 0)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate using LLM-as-a-Judge approach"
    )
    
    # Input: either model path OR response directory
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to model to evaluate (not needed if using --response_dir)")
    parser.add_argument("--response_dir", type=str, default=None,
                       help="Directory with pre-collected responses from run_inference_benchmarks.py")
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model name (required for LoRA)")
    parser.add_argument("--is_base", action="store_true",
                       help="Evaluate base model")
    
    # Judge model options
    parser.add_argument("--judge_model", type=str, default=None,
                       help="Path to judge model (default: use same as evaluated model)")
    parser.add_argument("--judge_base_model", type=str, default=None,
                       help="Judge base model (for LoRA)")
    
    # API options for evaluated model
    parser.add_argument("--use_api", action="store_true",
                       help="Use API for model being evaluated")
    parser.add_argument("--api_endpoint", type=str,
                       help="API endpoint for evaluated model")
    parser.add_argument("--api_model", type=str,
                       help="API model name for evaluated model")
    parser.add_argument("--api_key", type=str, default="",
                       help="API key")
    
    # API options for judge
    parser.add_argument("--judge_use_api", action="store_true",
                       help="Use API for judge")
    parser.add_argument("--judge_api_endpoint", type=str,
                       help="API endpoint for judge")
    parser.add_argument("--judge_api_model", type=str,
                       help="API model name for judge")
    parser.add_argument("--judge_api_key", type=str, default="",
                       help="Judge API key")
    
    # vLLM options for judge
    parser.add_argument("--judge_use_vllm", action="store_true",
                       help="Use vLLM for fast batch judge inference (requires vllm package)")
    parser.add_argument("--judge_gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory fraction to use with vLLM judge (0.0-1.0, default: 0.9)")
    parser.add_argument("--judge_batch_size", type=int, default=16,
                       help="Batch size for vLLM judge inference (default: 16)")
    
    # Evaluation options
    parser.add_argument("--tasks", nargs="+", 
                       default=["mcq", "rcm", "vsp", "ate"],
                       help="Tasks to evaluate")
    parser.add_argument("--output", type=str, default=None,
                       help="Base name for output directory (e.g., 'eval_results_judge_azure'). All results will be saved in '{output}_detailed/' directory including results.json, summary.json, and detailed JSONL files. Do not include .json extension.")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit samples per task")
    
    args = parser.parse_args()
    
    # Validate input arguments
    if not args.model_path and not args.response_dir:
        parser.error("Either --model_path or --response_dir must be provided")
    
    if args.response_dir and args.model_path:
        parser.error("Cannot specify both --model_path and --response_dir. Use --response_dir to evaluate pre-collected responses.")
    
    print("\n" + "="*70)
    print("LLM-AS-A-JUDGE EVALUATION")
    print("="*70)
    
    if args.response_dir:
        print(f"Mode: Loading pre-collected responses from {args.response_dir}")
        print(f"Tasks will be discovered from available response files")
    else:
        print(f"Mode: Running inference on model {args.model_path}")
    print("="*70)
    
    # Load model to evaluate (only if not using response_dir)
    if args.response_dir:
        # Skip model loading when using pre-collected responses
        model = None
        tokenizer = None
        print(f"\nSkipping model loading (using pre-collected responses)")
    elif args.use_api:
        if not args.api_endpoint or not args.api_model:
            parser.error("--api_endpoint and --api_model required with --use_api")
        model = None
        tokenizer = None
        print(f"\nEvaluating API model: {args.api_model}")
    else:
        if not args.model_path:
            parser.error("--model_path required when not using --response_dir")
        print(f"\nLoading model to evaluate: {args.model_path}")
        model, tokenizer = load_model_and_tokenizer(
            args.model_path, args.base_model, args.is_base
        )
    
    # Load judge model
    judge_vllm = None
    if args.judge_use_vllm:
        if not args.judge_model:
            parser.error("--judge_model required when using --judge_use_vllm")
        print(f"\nInitializing vLLM judge with model: {args.judge_model}")
        judge_vllm = initialize_judge_vllm(
            args.judge_model,
            args.judge_gpu_memory_utilization
        )
        judge_model = None
        judge_tokenizer = None
    elif args.judge_model:
        print(f"\nLoading judge model: {args.judge_model}")
        judge_model, judge_tokenizer = load_model_and_tokenizer(
            args.judge_model, args.judge_base_model, False
        )
    elif args.judge_use_api:
        if not args.judge_api_endpoint or not args.judge_api_model:
            parser.error("--judge_api_endpoint and --judge_api_model required")
        judge_model = None
        judge_tokenizer = None
        print(f"\nUsing API judge model: {args.judge_api_model}")
    else:
        # Use same model as judge
        judge_model = model
        judge_tokenizer = tokenizer
        print("\nUsing same model as judge")
    
    # Generate output filename
    if args.output is None:
        if args.response_dir:
            # Extract model name from response directory
            model_name = os.path.basename(args.response_dir.rstrip('/'))
        else:
            model_name = (args.api_model if args.use_api 
                         else args.model_path).rstrip('/').split('/')[-1]
            model_name = model_name.replace('/', '-').replace('\\', '-')
        args.output = f"eval_llm_judge_{model_name}.json"
    
    # Create detailed output directory
    detailed_dir = args.output.replace('.json', '_detailed')
    os.makedirs(detailed_dir, exist_ok=True)
    print(f"\nDetailed results will be saved to: {detailed_dir}/")
    
    # Prepare API kwargs
    model_api_kwargs = {
        'use_api': args.use_api,
        'api_endpoint': args.api_endpoint,
        'api_model': args.api_model,
        'api_key': args.api_key
    }
    
    judge_api_kwargs = {
        'use_api': args.judge_use_api,
        'api_endpoint': args.judge_api_endpoint,
        'api_model': args.judge_api_model,
        'api_key': args.judge_api_key
    }
    
    # Load metadata from response directory if available
    response_metadata = {}
    if args.response_dir:
        metadata_path = os.path.join(args.response_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    response_metadata = json.load(f)
                print(f"\nLoaded metadata from: {metadata_path}")
            except Exception as e:
                print(f"\nWarning: Could not load metadata from {metadata_path}: {e}")
    
    # Run evaluations
    results = {
        "model_path": response_metadata.get("model_path") or args.model_path,
        "judge_model": args.judge_model or args.judge_api_model or "same",
        "judge_api": args.judge_use_api,
        "is_base_model": args.is_base,
        "tasks": {}
    }
    
    # Determine tasks to evaluate
    if args.response_dir:
        # Auto-discover tasks from response directory
        import glob
        response_files = glob.glob(os.path.join(args.response_dir, "*_responses.jsonl"))
        discovered_tasks = [os.path.basename(f).replace('_responses.jsonl', '').replace('-', '_') 
                          for f in response_files]
        
        # If user specified tasks, filter to those; otherwise use all discovered
        if args.tasks and args.tasks != ["mcq", "rcm", "vsp", "ate"]:  # not default
            tasks_to_eval = [t for t in args.tasks if t.lower() in discovered_tasks]
            missing = [t for t in args.tasks if t.lower() not in discovered_tasks]
            if missing:
                print(f"\\nWarning: Tasks not found in response_dir: {missing}")
        else:
            tasks_to_eval = discovered_tasks
        
        print(f"\\nDiscovered tasks: {discovered_tasks}")
        print(f"Evaluating tasks: {tasks_to_eval}")
    else:
        tasks_to_eval = args.tasks
    
    # Task mapping (only needed when loading datasets, not responses)
    # Format: "task_name": ("dataset_source", "subset_or_split")
    # For JSONL tasks from AthenaBench GitHub: use "jsonl" as first element
    task_datasets = {
        # RISys-Lab HuggingFace datasets (cleaned AthenaBench 4 original tasks)
        "mcq": ("RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-mcq"),
        "rcm": ("RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-rcm"),
        "vsp": ("RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-vsp"),
        "ate": ("RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-ate"),
        "cti_taa": ("RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-taa"),  # Original TAA: 50 items
        
        # Other HuggingFace benchmarks
        "mmlu-cs": ("lighteval/mmlu", "computer_security"),
        "secure_maet": ("RISys-Lab/Benchmarks_CyberSec_SECURE", "MAET"),
        "secure_cwet": ("RISys-Lab/Benchmarks_CyberSec_SECURE", "CWET"),
        "secure_kcv": ("RISys-Lab/Benchmarks_CyberSec_SECURE", "KCV"),
        "secbench": ("RISys-Lab/Benchmarks_CyberSec_SecBench", "MCQs_English"),
        "cybermetric": ("RISys-Lab/Benchmarks_CyberSec_CyberMetrics", "cyberMetric_500"),
        
        # RedSageMCQ (5 subsets, 30K total samples)
        "redsage_frameworks": ("RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_knowledge_frameworks"),
        "redsage_generals": ("RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_knowledge_generals"),
        "redsage_skills": ("RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_skills"),
        "redsage_cli": ("RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_tools_cli"),
        "redsage_kali": ("RISys-Lab/Benchmarks_CyberSec_RedSageMCQ", "cybersecurity_tools_kali"),
        
        # AthenaBench GitHub JSONL tasks (3 additional tasks)
        "ckt": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-ckt-3k.jsonl"),
        "rms": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-rms.jsonl"),
        "taa": ("jsonl", "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark/athena-cti-taa.jsonl"),
    }
    
    for task_name in tasks_to_eval:
        task_lower = task_name.lower()
        
        print(f"\n{'='*70}")
        print(f"Evaluating {task_lower.upper()}")
        print(f"{'='*70}")
        
        detailed_output = os.path.join(detailed_dir, f"{task_lower}_detailed.jsonl")
        
        # Branch: Load from response_dir OR run inference on dataset
        if args.response_dir:
            # Load from pre-collected responses
            # Try both underscore and hyphen variants for compatibility
            response_file = os.path.join(args.response_dir, f"{task_lower}_responses.jsonl")
            if not os.path.exists(response_file):
                # Fallback to hyphen variant for backward compatibility
                response_file_hyphen = os.path.join(args.response_dir, f"{task_lower.replace('_', '-')}_responses.jsonl")
                if os.path.exists(response_file_hyphen):
                    response_file = response_file_hyphen
                else:
                    print(f"Skipping {task_lower}: response file not found ({response_file})")
                    continue
            
            print(f"Loading responses from: {response_file}")
            
            # Load JSONL responses
            responses = []
            with open(response_file, 'r') as f:
                for line in f:
                    responses.append(json.loads(line))
            
            if args.max_samples:
                responses = responses[:args.max_samples]
            
            print(f"Loaded {len(responses)} responses")
            
            # Determine task type from metadata (generic approach)
            # Read task_type from first response's metadata if available
            if responses and 'metadata' in responses[0] and 'task_type' in responses[0]['metadata']:
                judge_task_type = responses[0]['metadata']['task_type']
                print(f"Task type from metadata: {judge_task_type}")
            else:
                # Fallback to hard-coded mapping for backward compatibility
                judge_task_type_map = {
                    "secure_maet": "secure",
                    "secure_cwet": "secure",
                    "secure_kcv": "secure",
                    "secbench": "secbench",
                    "mmlu-cs": "mmlu_cs",
                    "ckt": "ckt",
                    "rms": "rms",
                    "taa": "taa",
                    "cti_taa": "taa",
                    "redsage_frameworks": "mcq",
                    "redsage_generals": "mcq",
                    "redsage_skills": "mcq",
                    "redsage_cli": "mcq",
                    "redsage_kali": "mcq",
                }
                judge_task_type = judge_task_type_map.get(task_lower, task_lower)
                print(f"Task type from fallback mapping: {judge_task_type}")
            
            # Convert to dataset-like format
            dataset_dict = {
                'Prompt': [r.get('prompt', r.get('question', '')) for r in responses],
                'GT': [r['ground_truth'] for r in responses],
                'model_response': [r['model_response'] for r in responses],
                'metadata': [r.get('metadata', {}) for r in responses]
            }
            dataset = HFDataset.from_dict(dataset_dict)
            
            # Evaluate using pre-collected responses
            task_results = evaluate_with_judge_from_responses(
                dataset, judge_task_type,
                judge_model, judge_tokenizer, judge_vllm=judge_vllm,
                detailed_output=detailed_output,
                **judge_api_kwargs
            )
        
        elif task_lower in task_datasets:
            # Load dataset and run inference
            dataset_name, subset_name = task_datasets[task_lower]
            
            # Determine task type from task name (for dataset mode only)
            judge_task_type_map = {
                "secure_maet": "secure",
                "secure_cwet": "secure",
                "secure_kcv": "secure",
                "secbench": "secbench",
                "mmlu-cs": "mmlu_cs",
                "ckt": "ckt",
                "rms": "rms",
                "taa": "taa",
                "cti_taa": "taa",
                "redsage_frameworks": "mcq",
                "redsage_generals": "mcq",
                "redsage_skills": "mcq",
                "redsage_cli": "mcq",
                "redsage_kali": "mcq",
            }
            judge_task_type = judge_task_type_map.get(task_lower, task_lower)
            
            # Load dataset based on type
            if dataset_name == "jsonl":
                print(f"Loading JSONL dataset from: {subset_name}")
                dataset = load_jsonl_dataset(subset_name)
            else:
                print(f"Loading dataset: {dataset_name}/{subset_name}")
                dataset = load_dataset(dataset_name, subset_name, split="test")
            
            if args.max_samples:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            
            # Run inference and judge
            task_results = evaluate_with_judge(
                model, tokenizer, dataset, judge_task_type,
                judge_model, judge_tokenizer, judge_vllm=judge_vllm,
                detailed_output=detailed_output,
                **model_api_kwargs
            )
        else:
            print(f"Unknown task: {task_lower}")
            continue
        
        # Store and print results
        results["tasks"][task_lower.upper()] = task_results
        
        # Print results
        print(f"\nResults for {task_lower.upper()}:")
        if "mad" in task_results:
            # VSP - regression task
            print(f"  MAD: {task_results['mad']:.4f}")
            print(f"  Total: {task_results['total']}")
        elif "precision" in task_results:
            # ATE and other metric-based tasks
            print(f"  Precision: {task_results['precision']:.4f}")
            print(f"  Recall: {task_results['recall']:.4f}")
            print(f"  F1: {task_results['f1']:.4f}")
        else:
            # Classification tasks
            print(f"  Accuracy: {task_results['accuracy']:.4f}")
            print(f"  Correct: {task_results['correct']}/{task_results['total']}")
    
    # Save final results and generate summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(json.dumps(results, indent=2))
    
    # Save all JSON results inside the detailed results directory
    results_json_path = os.path.join(detailed_dir, 'results.json')
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_json_path}")
    print(f"Detailed results directory: {detailed_dir}/")
    
    # Generate and save clean summary from detailed JSONL results
    print("\n" + "="*70)
    print("GENERATING SUMMARY FROM DETAILED RESULTS")
    print("="*70)
    
    try:
        summary = generate_summary_from_detailed(detailed_dir)
        
        # Save summary inside the same directory
        summary_json_path = os.path.join(detailed_dir, 'summary.json')
        with open(summary_json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary generated from detailed results:")
        print(f"  Overall Accuracy: {summary['overall_accuracy']:.1%}")
        print(f"  Total Correct: {summary['total_correct']}/{summary['total_samples']}")
        print(f"  Tasks Evaluated: {len(summary['tasks'])}")
        print(f"\nSummary saved to: {summary_json_path}")
        print(f"\n✓ All results organized in: {detailed_dir}/")
        
    except Exception as e:
        print(f"\nWarning: Could not generate summary from detailed results: {e}")


if __name__ == "__main__":
    main()
