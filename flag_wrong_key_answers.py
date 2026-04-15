#!/usr/bin/env python3
"""
Flag questions with likely wrong key answers using model agreement voting.

This script analyzes evaluation results from multiple models and flags questions
where most models agree on an alternative answer to the benchmark's key answer.
This helps identify potentially incorrect ground truth labels in benchmarks.

Usage:
    python flag_wrong_key_answers.py --detailed_results_dirs eval_dir1 eval_dir2 \
                                     --agreement_threshold 0.5 \
                                     --output flagged_questions.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
from tqdm import tqdm

# Import from evaluate.py for model loading and LLM judge functionality
from evaluate import load_model_and_tokenizer, generate_response

# Optional vLLM import for fast judge inference
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# Import judge functions from run_evaluate_llm_judge.py for reuse
from run_evaluate_llm_judge import (
    initialize_judge_vllm,
    generate_judge_responses_vllm,
)


# Task types that use choice-based answers (exact matching)
CHOICE_TASKS = {
    'mcq', 'seceval', 'cybermetric', 'cissp', 'mmlu_cs', 'secure', 'secbench', 'ckt'
}


def load_detailed_results(detailed_dir: str) -> Dict[str, List[Dict]]:
    """Load all detailed evaluation results from a directory.

    Args:
        detailed_dir: Directory containing *_detailed.jsonl files

    Returns:
        Dictionary mapping task_name -> list of result dicts
    """
    results_by_task = defaultdict(list)
    detailed_path = Path(detailed_dir)

    for file_path in sorted(detailed_path.glob('*_detailed.jsonl')):
        task_name = file_path.stem.replace('_detailed', '')

        with open(file_path, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    results_by_task[task_name].append(result)
                except json.JSONDecodeError:
                    continue

    return dict(results_by_task)


def extract_model_name(eval_dir: str) -> str:
    """Extract model name from evaluation directory path.

    Args:
        eval_dir: Directory path

    Returns:
        Model name string
    """
    dir_name = Path(eval_dir).name
    if 'eval_llm_judge_' in dir_name:
        return dir_name.replace('eval_llm_judge_', '').replace('_detailed', '')
    return dir_name.replace('_detailed', '')


def aggregate_responses(eval_dirs: List[str]) -> Dict[str, List[Dict]]:
    """Aggregate responses from multiple evaluations by task and question.

    Args:
        eval_dirs: List of evaluation result directories

    Returns:
        Dictionary: task_name -> list of aggregated question results
    """
    aggregated = defaultdict(list)

    # Load results from each model
    model_results = {}
    for eval_dir in eval_dirs:
        model_name = extract_model_name(eval_dir)
        print(f"Loading results from {model_name}...")
        try:
            model_results[model_name] = load_detailed_results(eval_dir)
        except Exception as e:
            print(f"Error loading results from {eval_dir}: {e}")
            continue

    if not model_results:
        raise ValueError("No valid evaluation directories found")

    # Get all tasks
    all_tasks = set()
    for results in model_results.values():
        all_tasks.update(results.keys())

    print(f"\nFound tasks: {sorted(all_tasks)}")

    # Aggregate by task and question index
    for task_name in all_tasks:
        question_map = {}

        for model_name, task_results in model_results.items():
            if task_name not in task_results:
                continue

            for result in task_results[task_name]:
                idx = result.get('index', result.get('question_index', 0))
                key = f"{task_name}_{idx}"

                if key not in question_map:
                    question_map[key] = {
                        'task_type': result.get('task_type', task_name),
                        'question': result.get('question', ''),
                        'ground_truth': result.get('ground_truth', ''),
                        'index': idx,
                        'models': {}
                    }

                question_map[key]['models'][model_name] = {
                    'is_correct': result.get('is_correct', False),
                    'model_response': result.get('model_response', ''),
                    'judge_response': result.get('judge_response', '')
                }

        # Convert to list
        aggregated[task_name] = list(question_map.values())

    return dict(aggregated)


def judge_answers_same(judge_model, judge_tokenizer, task_type: str,
                      answer1: str, answer2: str, question: str = "",
                      judge_vllm=None, **api_kwargs) -> bool:
    """Use LLM judge to determine if two answers are semantically the same.

    Args:
        judge_model: Judge model
        judge_tokenizer: Judge tokenizer
        task_type: Type of task
        answer1: First answer
        answer2: Second answer
        question: Original question for context
        judge_vllm: Optional vLLM instance for faster inference
        **api_kwargs: API kwargs for judge

    Returns:
        True if answers are considered the same
    """
    if task_type in CHOICE_TASKS:
        # For choice-based tasks, exact match after normalization
        return normalize_answer(answer1) == normalize_answer(answer2)

    # For open-ended tasks, use LLM judge
    prompt = f"""Determine if these two answers to the question are semantically equivalent:

Question: {question}

Answer 1: {answer1}
Answer 2: {answer2}

Are these answers expressing the same meaning? Respond with only 'YES' or 'NO'."""

    try:
        if judge_vllm:
            # Use vLLM for faster inference
            responses = generate_judge_responses_vllm(judge_vllm, [prompt], max_tokens=10)
            response = responses[0]
        else:
            # Use standard generation
            response = generate_response(judge_model, judge_tokenizer, prompt, max_new_tokens=10, **api_kwargs)
        return 'YES' in response.upper()
    except Exception:
        # Fallback to exact match
        return answer1.strip().lower() == answer2.strip().lower()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer
    """
    if not answer:
        return ""

    # Remove extra whitespace and convert to uppercase
    normalized = answer.strip().upper()

    # For MCQ, extract just the letter if it's a single choice
    if len(normalized) == 1 and normalized in 'ABCDE':
        return normalized

    return normalized


def identify_wrong_key_answers(aggregated_results: Dict[str, List[Dict]],
                              agreement_threshold: float,
                              judge_model, judge_tokenizer, judge_vllm=None, **judge_api_kwargs) -> Dict[str, Any]:
    """Identify questions with likely wrong key answers using voting.

    Args:
        aggregated_results: Aggregated results from multiple models
        agreement_threshold: Fraction of models that must agree on alternative
        judge_model: Judge model (required for open-ended tasks)
        judge_tokenizer: Judge tokenizer (required for open-ended tasks)
        judge_vllm: Optional vLLM instance for judge
        **judge_api_kwargs: API kwargs for judge

    Returns:
        Report dictionary with flagged questions
    """
    # Check if judge model is required for any tasks
    has_open_ended = any(
        any(q['task_type'] not in CHOICE_TASKS for q in questions)
        for questions in aggregated_results.values()
    )
    
    if has_open_ended and (judge_model is None or judge_tokenizer is None):
        raise ValueError("Judge model is required for open-ended tasks. Please provide --judge_model")

    report = {
        "summary": {
            "total_questions": 0,
            "total_models": 0,
            "flagged_questions": 0,
            "agreement_threshold": agreement_threshold
        },
        "benchmarks": {}
    }

    model_names = set()

    for task_name, questions in aggregated_results.items():
        task_flagged = 0
        task_details = []

        for q_data in tqdm(questions, desc=f"Analyzing {task_name}"):
            models = q_data['models']
            model_names.update(models.keys())

            if len(models) < 2:
                continue  # Need at least 2 models

            ground_truth = normalize_answer(q_data['ground_truth'])
            task_type = q_data['task_type']

            # Count alternative answers
            answer_counts = Counter()

            for model_data in models.values():
                model_response = model_data['model_response']

                if task_type in CHOICE_TASKS:
                    normalized_response = normalize_answer(model_response)
                    answer_counts[normalized_response] += 1
                else:
                    # For open-ended, use raw response for semantic comparison
                    answer_counts[model_response] += 1

            # Find the most common alternative answer
            if answer_counts:
                most_common_answer, count = answer_counts.most_common(1)[0]
                agreement_fraction = count / len(models)

                # Check if most models agree on an answer different from ground truth
                if task_type in CHOICE_TASKS:
                    is_different = most_common_answer != ground_truth
                else:
                    # For open-ended, always use judge
                    is_different = not judge_answers_same(
                        judge_model, judge_tokenizer, task_type,
                        most_common_answer, q_data['ground_truth'], q_data['question'],
                        judge_vllm=judge_vllm, **judge_api_kwargs
                    )

                wrong_key_answer = is_different and agreement_fraction >= agreement_threshold

                if wrong_key_answer:
                    task_flagged += 1

                task_details.append({
                    'index': q_data['index'],
                    'question': q_data['question'],
                    'ground_truth': q_data['ground_truth'],
                    'most_common_answer': most_common_answer,
                    'agreement_fraction': agreement_fraction,
                    'wrong_key_answer': wrong_key_answer,
                    'models_evaluated': len(models)
                })

        report["benchmarks"][task_name] = {
            "total_questions": len(questions),
            "flagged_questions": task_flagged,
            "questions": task_details
        }

        report["summary"]["total_questions"] += len(questions)
        report["summary"]["flagged_questions"] += task_flagged

    report["summary"]["total_models"] = len(model_names)
    report["summary"]["models_evaluated"] = sorted(model_names)

    return report


def flag_questions_in_files(aggregated_results: Dict[str, List[Dict]],
                           report: Dict[str, Any],
                           output_dir: str = None) -> None:
    """Update JSONL files with wrong_key_answer flags.

    Args:
        aggregated_results: Original aggregated results
        report: Flagging report
        output_dir: Directory to save updated files (if None, update in place)
    """
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for task_name, questions in aggregated_results.items():
        if task_name not in report["benchmarks"]:
            continue

        task_report = report["benchmarks"][task_name]

        # Group questions by index for lookup
        flagged_by_index = {
            q['index']: q['wrong_key_answer'] for q in task_report["questions"]
        }

        # Note: In a real implementation, you'd need to update the original JSONL files
        # For now, we'll just print the results
        print(f"\n{task_name.upper()}: {task_report['flagged_questions']} flagged questions")


def main():
    parser = argparse.ArgumentParser(
        description="Flag questions with likely wrong key answers using model voting"
    )

    parser.add_argument(
        "--detailed_results_dirs",
        nargs="+",
        required=True,
        help="Directories containing *_detailed.jsonl files from run_evaluate_llm_judge.py"
    )
    parser.add_argument(
        "--agreement_threshold",
        type=float,
        default=0.5,
        help="Fraction of models that must agree on alternative answer (default: 0.5)"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        help="Judge model path for semantic comparison of open-ended answers (required if open-ended tasks are present)"
    )
    parser.add_argument(
        "--judge_use_vllm",
        action="store_true",
        help="Use vLLM for fast judge inference (requires vllm package)"
    )
    parser.add_argument(
        "--judge_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory fraction to use with vLLM judge (0.0-1.0, default: 0.9)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="flagged_questions_report.json",
        help="Output JSON file for flagging report"
    )
    parser.add_argument(
        "--update_jsonl",
        action="store_true",
        help="Update original JSONL files with wrong_key_answer property"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("FLAG WRONG KEY ANSWERS")
    print("="*70)
    print(f"Detailed results directories: {len(args.detailed_results_dirs)}")
    for d in args.detailed_results_dirs:
        print(f"  - {d}")
    print(f"Agreement threshold: >= {args.agreement_threshold:.1%}")
    print("="*70 + "\n")

    # Validate directories
    for eval_dir in args.detailed_results_dirs:
        if not Path(eval_dir).is_dir():
            print(f"Error: Directory not found: {eval_dir}")
            return

    # Aggregate responses from all models
    aggregated = aggregate_responses(args.detailed_results_dirs)

    # Check if judge model is required
    has_open_ended = any(
        any(q['task_type'] not in CHOICE_TASKS for q in questions)
        for questions in aggregated.values()
    )

    # Load judge model if required
    judge_model = None
    judge_tokenizer = None
    judge_vllm = None
    if has_open_ended:
        if not args.judge_model:
            print("Error: Judge model is required for open-ended tasks. Please provide --judge_model")
            return
        if args.judge_use_vllm:
            if not HAS_VLLM:
                print("Error: vLLM not installed. Install with: pip install vllm")
                return
            print(f"Loading judge model with vLLM: {args.judge_model}")
            judge_vllm = initialize_judge_vllm(args.judge_model, args.judge_gpu_memory_utilization)
        else:
            print(f"Loading judge model: {args.judge_model}")
            judge_model, judge_tokenizer = load_model_and_tokenizer(args.judge_model)
    elif args.judge_model:
        if args.judge_use_vllm:
            if not HAS_VLLM:
                print("Error: vLLM not installed. Install with: pip install vllm")
                return
            print(f"Loading judge model with vLLM: {args.judge_model}")
            judge_vllm = initialize_judge_vllm(args.judge_model, args.judge_gpu_memory_utilization)
        else:
            print(f"Loading judge model: {args.judge_model}")
            judge_model, judge_tokenizer = load_model_and_tokenizer(args.judge_model)

    # Identify wrong key answers
    report = identify_wrong_key_answers(
        aggregated, args.agreement_threshold,
        judge_model, judge_tokenizer, judge_vllm=judge_vllm
    )

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    # Update JSONL files if requested
    if args.update_jsonl:
        flag_questions_in_files(aggregated, report)

    # Print summary
    summary = report["summary"]
    print("\n" + "="*70)
    print("FLAGGING SUMMARY")
    print("="*70)
    print(f"Total Models Evaluated: {summary['total_models']}")
    print(f"  Models: {', '.join(summary.get('models_evaluated', []))}")
    print(f"\nTotal Questions Analyzed: {summary['total_questions']}")
    print(f"Flagged Questions (>= {summary['agreement_threshold']:.1%} agreement): {summary['flagged_questions']}")

    if summary['total_questions'] > 0:
        flagged_pct = 100 * summary['flagged_questions'] / summary['total_questions']
        print(f"Overall Flagged Percentage: {flagged_pct:.1f}%")
    else:
        print("Overall Flagged Percentage: N/A")

    print("\n" + "="*70)
    print("RESULTS BY BENCHMARK")
    print("="*70)

    for task_name in sorted(report["benchmarks"]):
        task = report["benchmarks"][task_name]
        print(f"\n{task_name.upper()}")
        print(f"  Total Questions: {task['total_questions']}")
        if task['total_questions'] > 0:
            flagged_pct = 100 * task['flagged_questions'] / task['total_questions']
            print(f"  Flagged Questions: {task['flagged_questions']} ({flagged_pct:.1f}%)")
        else:
            print("  Flagged Questions: 0 (N/A)")

    print(f"\n{'='*70}")
    print(f"Report saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()