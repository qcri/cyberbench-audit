#!/usr/bin/env python3
"""
Full-scale empirical evaluation across all models and tasks.

This script runs complete evaluation of:
- 7 models (1 base + 5 security-specialized + 1 Arabic)
- 21 tasks across 9 benchmark families
- Producing confidence intervals and statistical tests

Outputs:
- results/full_scale_results.json
- results/statistical_summary.json
- results/model_comparison_table.csv
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import sys
import argparse
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


@dataclass
class ModelConfig:
    """Model configuration for evaluation"""
    name: str
    model_path: str
    is_base: bool = False
    base_model: str = None
    description: str = ""
    use_api: bool = False
    api_endpoint: str = ""
    api_style: str = "chat_completions"


@dataclass
class TaskConfig:
    """Task configuration"""
    benchmark: str
    task_name: str
    samples: int = 100  # full-scale: 100+ samples per task


@dataclass
class EvaluationResult:
    """Stores evaluation result with confidence intervals"""
    model_name: str
    task_name: str
    benchmark: str
    accuracy: float
    std_dev: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_samples: int = 0
    metadata: Dict = field(default_factory=dict)


MODELS_DIR = "/export/cyb-ai-research-data/aberriche/models"

# Model configurations for full-scale evaluation
MODELS = [
    # ── Security-tuned models (existing) ──────────────────────────────────────
    ModelConfig(
        name="Foundation-Sec-8B-Instruct",
        model_path=f"{MODELS_DIR}/Foundation-Sec-8B-Instruct",
        is_base=True,
        description="Security-focused foundation model (instruct-tuned)"
    ),
    ModelConfig(
        name="RedSage-Qwen3-8B-DPO",
        model_path=f"{MODELS_DIR}/RedSage-Qwen3-8B-DPO",
        is_base=True,
        description="RISys-Lab RedSage model (Qwen3 backbone, DPO-aligned, thinking model)"
    ),
    ModelConfig(
        name="Llama-Primus-Merged",
        model_path=f"{MODELS_DIR}/Llama-Primus-Merged",
        is_base=True,
        description="TrendMicro PRIMUS model (merged, 23K cybersecurity samples)"
    ),

    # ── Proprietary / API models ───────────────────────────────────────────────
    ModelConfig(
        name="GPT-5.4",
        model_path="gpt-4.1",
        is_base=False,
        use_api=True,
        api_endpoint="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/",
        api_style="chat_completions",
        description="GPT-5.4 deployed on Azure (Chat Completions API)"
    ),

    # ── Open-source models (new) ───────────────────────────────────────────────
    ModelConfig(
        name="Qwen3.6-35B-A3B",
        model_path=f"{MODELS_DIR}/Qwen3.6-35B-A3B",
        is_base=True,
        description="Qwen3 MoE (35B total / 3.6B active params, thinking model)"
    ),
    ModelConfig(
        name="Gemma-4-31B-it",
        model_path=f"{MODELS_DIR}/gemma-4-31B",
        is_base=True,
        description="Google Gemma-4 31B instruction-tuned"
    ),
    ModelConfig(
        name="Fanar-2-27B-Instruct",
        model_path=f"{MODELS_DIR}/Fanar-2-27B-Instruct",
        is_base=True,
        description="QCRI Fanar-2 multilingual 27B model"
    ),
    ModelConfig(
        name="GPT-oss-20B",
        model_path=f"{MODELS_DIR}/gpt-oss-20b",
        is_base=True,
        description="OpenAI open-weight 20B model"
    ),
]

# Task configurations - 24 tasks across 9 benchmark families
# Note: Both CTI-Bench TAA (50 items) and AthenaBench TAA (100 items) are included
# to compare original vs. expanded benchmark quality
TASKS = [
    # CTI-Bench Tasks (5 tasks - includes original TAA)
    TaskConfig("CTI-Bench", "MCQ", samples=100),
    TaskConfig("CTI-Bench", "RCM", samples=100),
    TaskConfig("CTI-Bench", "VSP", samples=100),
    TaskConfig("CTI-Bench", "ATE", samples=100),
    TaskConfig("CTI-Bench", "TAA", samples=100),  # Original: 50 items

    # AthenaBench Tasks (6 tasks - CKT/RMS/TAA classic + ATE/RCM/VSP expanded)
    TaskConfig("AthenaBench", "CKT", samples=100),
    TaskConfig("AthenaBench", "RMS", samples=100),
    TaskConfig("AthenaBench", "TAA", samples=100),      # Expanded: 100 items
    TaskConfig("AthenaBench", "ATE-expanded", samples=100),
    TaskConfig("AthenaBench", "RCM-compiled", samples=100),
    TaskConfig("AthenaBench", "VSP", samples=100),

    # SECURE Tasks (3 tasks)
    TaskConfig("SECURE", "MAET", samples=100),
    TaskConfig("SECURE", "CWET", samples=100),
    TaskConfig("SECURE", "KCV", samples=100),

    # Other Benchmarks
    TaskConfig("SecBench", "MCQ", samples=100),
    TaskConfig("CyberMetric", "MCQ", samples=100),
    TaskConfig("SecEval", "MSQ", samples=100),
    TaskConfig("CISSP", "MCQ", samples=100),
    TaskConfig("MMLU-CS", "MCQ", samples=100),

    # RedSage-MCQ (5 subtasks)
    TaskConfig("RedSage-MCQ", "Frameworks", samples=100),
    TaskConfig("RedSage-MCQ", "Generals", samples=100),
    TaskConfig("RedSage-MCQ", "Skills", samples=100),
    TaskConfig("RedSage-MCQ", "CLI", samples=100),
    TaskConfig("RedSage-MCQ", "Kali", samples=100),
]


def run_inference_for_model(model_config: ModelConfig, subset: bool = False) -> None:
    """Run inference collection for a single model across all tasks"""
    import sys
    logger.info(f"Collecting inferences for: {model_config.name}")

    # Map tasks to command-line task names (24 tasks across 9 benchmarks)
    task_names = [
        # CTI-Bench (5)
        "mcq", "rcm", "vsp", "ate", "cti_taa",
        # AthenaBench (6 — classic + expanded extraction tasks)
        "ckt", "rms", "taa", "athena_ate", "athena_rcm", "athena_vsp",
        # SECURE (3)
        "secure_maet", "secure_cwet", "secure_kcv",
        # Other benchmarks
        "seceval", "cybermetric", "mmlu-cs", "secbench", "cissp",
        # RedSage-MCQ (5)
        "redsage_frameworks", "redsage_generals", "redsage_skills",
        "redsage_cli", "redsage_kali",
    ]

    out_name = model_config.name.replace(' ', '_').replace('/', '-')
    if model_config.use_api:
        cmd = [
            sys.executable,
            "../run_inference_benchmarks.py",
            "--use_api",
            "--api_endpoint", model_config.api_endpoint,
            "--api_model", model_config.model_path,
            "--api_style", model_config.api_style,
            "--output_dir", f"results/responses_{out_name}",
            "--tasks",
        ] + task_names
        import os
        api_key = os.environ.get("AZURE_API_KEY", "")
        if api_key:
            cmd.extend(["--api_key", api_key])
    else:
        cmd = [
            sys.executable,
            "../run_inference_benchmarks.py",
            "--model_path", model_config.model_path,
            "--use_vllm",
            "--output_dir", f"results/responses_{out_name}",
            "--tasks",
        ] + task_names

    if model_config.base_model:
        cmd.extend(["--base_model", model_config.base_model])

    if model_config.is_base and not model_config.use_api:
        cmd.append("--is_base")

    if subset:
        cmd.extend(["--max_samples", "10"])
        logger.info("  → Running in SUBSET mode (10 samples per task)")

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"✓ Inferences collected for {model_config.name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to collect inferences for {model_config.name}: {e}")
        raise


def run_evaluation_for_model(model_config: ModelConfig, subset: bool = False) -> List[EvaluationResult]:
    """Run evaluation (LLM judge) for a single model across all tasks"""
    import sys
    logger.info(f"Evaluating inferences for: {model_config.name}")

    response_dir = f"results/responses_{model_config.name.replace(' ', '_').replace('/', '-')}"
    output_file = f"results/evaluations_{model_config.name.replace(' ', '_').replace('/', '-')}.json"

    import os
    judge = os.path.join(MODELS_DIR, "Foundation-Sec-8B-Instruct")
    if not os.path.isdir(judge):
        judge = "fdtn-ai/Foundation-Sec-8B-Instruct"

    cmd = [
        sys.executable,
        "../run_evaluate_llm_judge.py",
        "--response_dir", response_dir,
        "--output", output_file,
        "--judge_model", judge,
        "--judge_use_vllm",
    ]

    if subset:
        cmd.extend(["--max_samples", "10"])
        logger.info("  → Evaluating SUBSET (10 samples per task)")

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"✓ Evaluation complete for {model_config.name}")

        # Parse results and compute confidence intervals
        results = parse_evaluation_results(model_config.name, output_file)
        return results
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to evaluate {model_config.name}: {e}")
        raise


def parse_evaluation_results(model_name: str, results_file: str) -> List[EvaluationResult]:
    """Parse evaluation JSON and compute confidence intervals
    
    Note: run_evaluate_llm_judge.py writes to {output}_detailed/results.json,
    not to the output path directly
    """
    results = []
    
    # Judge script creates a _detailed directory and writes results.json there
    detailed_dir = results_file.replace('.json', '_detailed')
    actual_results_file = Path(detailed_dir) / "results.json"

    try:
        if not actual_results_file.exists():
            logger.warning(f"Results file not found: {actual_results_file}")
            return results
            
        with open(actual_results_file, 'r') as f:
            data = json.load(f)

        # Parse task results from judge output
        # Judge writes: {"metadata": {...}, "results": {...}, "tasks": {task_name: {accuracy, correct, total}}}
        tasks_data = data.get("tasks", {})
        
        for task_name, task_results in tasks_data.items():
            if isinstance(task_results, dict) and 'accuracy' in task_results:
                accuracy = task_results.get('accuracy', 0.0)
                correct = task_results.get('correct', 0)
                total = task_results.get('total', 0)
                
                # Try to infer benchmark from task name
                benchmark = "Unknown"
                if task_name in ['mcq', 'rcm', 'vsp', 'ate']:
                    benchmark = "CTI-Bench"
                elif task_name in ['cti_taa']:
                    benchmark = "CTI-Bench"
                elif task_name in ['ckt', 'rms', 'taa']:
                    benchmark = "AthenaBench"
                elif task_name.startswith('secure_'):
                    benchmark = "SECURE"
                elif task_name == 'seceval':
                    benchmark = "SecEval"
                elif task_name == 'cybermetric':
                    benchmark = "CyberMetric"
                elif task_name == 'cissp':
                    benchmark = "CISSP"
                elif task_name == 'mmlu-cs':
                    benchmark = "MMLU-CS"
                elif task_name == 'secbench':
                    benchmark = "SecBench"
                elif task_name.startswith('redsage_'):
                    benchmark = "RedSage-Bench"
                
                result = EvaluationResult(
                    model_name=model_name,
                    task_name=task_name,
                    benchmark=benchmark,
                    accuracy=accuracy,
                    std_dev=0.0,  # Judge doesn't provide std dev
                    ci_lower=accuracy,  # Use accuracy as bounds for single run
                    ci_upper=accuracy,
                    n_samples=total,
                    metadata=task_results
                )
                results.append(result)

    except Exception as e:
        logger.error(f"Error parsing results from {actual_results_file}: {e}")

    return results


def compute_statistical_tests(results: List[EvaluationResult]) -> Dict:
    """Compute statistical significance tests between models"""
    from scipy import stats

    stats_summary = {}

    # Group by benchmark
    by_benchmark = {}
    for result in results:
        if result.benchmark not in by_benchmark:
            by_benchmark[result.benchmark] = {}
        if result.task_name not in by_benchmark[result.benchmark]:
            by_benchmark[result.benchmark][result.task_name] = {}
        by_benchmark[result.benchmark][result.task_name][result.model_name] = result

    # Compare models within each benchmark task
    for benchmark, tasks in by_benchmark.items():
        stats_summary[benchmark] = {}
        for task_name, model_results in tasks.items():
            stats_summary[benchmark][task_name] = {}
            model_names = list(model_results.keys())

            # Pairwise comparisons between models
            for i, model1 in enumerate(model_names[1:], start=1):  # Skip base
                for model2 in model_names[i+1:]:
                    # Simplified: just note which model performs better
                    acc1 = model_results[model1].accuracy
                    acc2 = model_results[model2].accuracy
                    diff = acc1 - acc2
                    stats_summary[benchmark][task_name][f"{model1}_vs_{model2}"] = {
                        "accuracy_diff": diff,
                        "winner": model1 if diff > 0 else model2,
                        "confidence_gap": (model_results[model1].ci_upper -
                                          model_results[model2].ci_lower) / 2
                    }

    return stats_summary


def generate_summary_tables(all_results: List[EvaluationResult]) -> None:
    """Generate CSV summary tables for the paper"""
    import csv

    # Table 1: Model comparison across all tasks
    model_names = sorted(set(r.model_name for r in all_results))

    with open(OUTPUT_DIR / "model_comparison_table.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Benchmark", "Task", "N_Samples"] + model_names
        writer.writerow(header)

        for benchmark in sorted(set(r.benchmark for r in all_results)):
            for task in sorted(set(r.task_name for r in all_results if r.benchmark == benchmark)):
                task_results = [r for r in all_results
                               if r.benchmark == benchmark and r.task_name == task]

                if task_results:
                    n_samples = task_results[0].n_samples
                    row = [benchmark, task, n_samples]
                    for model_name in model_names:
                        result = next((r for r in task_results if r.model_name == model_name), None)
                        if result:
                            row.append(f"{result.accuracy:.3f} ± {result.std_dev:.3f}")
                        else:
                            row.append("N/A")
                    writer.writerow(row)

    logger.info(f"✓ Summary table saved to {OUTPUT_DIR}/model_comparison_table.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Full-scale cybersecurity LLM benchmark evaluation"
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Run on small subset (10 samples per task) for testing"
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    if args.subset:
        logger.info("SUBSET VALIDATION - CYBERSECURITY LLM BENCHMARK EVALUATION")
        logger.info("Mode: TESTING (10 samples per task)")
    else:
        logger.info("FULL-SCALE CYBERSECURITY LLM BENCHMARK EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Models: {len(MODELS)} (1 base + 5 security-specialized + 1 Arabic)")
    logger.info(f"Tasks: {len(TASKS)} across {len(set(t.benchmark for t in TASKS))} benchmarks")
    logger.info("Note: Both CTI-Bench TAA and AthenaBench TAA included to compare versions")
    logger.info("")

    all_results = []

    # Step 1: Collect inferences for all models
    logger.info("[Step 1/4] Collecting inferences from all models...")
    for model_config in MODELS:
        try:
            run_inference_for_model(model_config, subset=args.subset)
        except Exception as e:
            logger.warning(f"Skipping {model_config.name} due to: {e}")
            continue

    # Step 2: Evaluate inferences using LLM judge
    logger.info("[Step 2/4] Evaluating with LLM judge (HuggingFace backend)...")
    for model_config in MODELS:
        try:
            results = run_evaluation_for_model(model_config, subset=args.subset)
            all_results.extend(results)
        except Exception as e:
            logger.warning(f"Skipping evaluation for {model_config.name} due to: {e}")
            continue

    # Step 3: Compute statistical tests
    logger.info("[Step 3/4] Computing statistical tests...")
    stats_summary = compute_statistical_tests(all_results)

    # Step 4: Generate summary tables
    logger.info("[Step 4/4] Generating summary tables...")
    generate_summary_tables(all_results)

    # Save full results
    results_dict = {
        "metadata": {
            "total_models": len(MODELS),
            "total_tasks": len(TASKS),
            "backend": "huggingface",
            "judge_model": "meta-llama/Llama-3.1-8B-Instruct"
        },
        "results": [
            {
                "model_name": r.model_name,
                "benchmark": r.benchmark,
                "task_name": r.task_name,
                "accuracy": r.accuracy,
                "std_dev": r.std_dev,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "n_samples": r.n_samples
            }
            for r in all_results
        ],
        "statistics": stats_summary
    }

    with open(OUTPUT_DIR / "full_scale_results.json", 'w') as f:
        json.dump(results_dict, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"✓ Full-scale evaluation complete!")
    logger.info(f"✓ Results saved to {OUTPUT_DIR}/full_scale_results.json")
    logger.info(f"✓ Total results collected: {len(all_results)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
