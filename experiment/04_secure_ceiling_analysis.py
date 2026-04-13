#!/usr/bin/env python3
"""
Test SECURE ceiling effect with multiple baseline models.

Hypothesis: SECURE MAET, CWET, KCV tasks saturate at 100% with
Llama-3.1-8B-Instruct (base model). This suggests ceiling effects rather
than true task difficulty.

This script tests multiple baselines of varying capability:
- Small models (7B, 8B non-specialized)
- Weak instruction-tuned baselines (generic LLMs)
- Strong instruction-tuned models (SOTA general models)
- Security-specialized models

If all models hit 100%, that strongly suggests the task is too easy.
If performance varies by model capability, the task is valid.

Output:
- results/secure_ceiling_analysis.json
- results/secure_model_performance.csv
- results/secure_ceiling_evidence.md
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


@dataclass
class ModelTier:
    """Model capability tier for baseline testing"""
    tier_name: str
    models: List[str]
    expected_difficulty: str  # Easy, Medium, Hard


@dataclass
class TaskResult:
    """Result for a single task"""
    task_name: str
    model_name: str
    accuracy: float
    n_samples: int = 0
    notes: str = ""


# Define model baseline tiers
BASELINE_TIERS = [
    ModelTier(
        tier_name="Weak Base Models",
        models=[
            "mistralai/Mistral-7B",
            "meta-llama/Llama-2-7B",
        ],
        expected_difficulty="Hard"
    ),
    ModelTier(
        tier_name="Standard Instruction-Tuned",
        models=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "google/gemma-2-9b-it",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        expected_difficulty="Medium"
    ),
    ModelTier(
        tier_name="Strong Instruction-Tuned",
        models=[
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.1-405B-Instruct",
        ],
        expected_difficulty="Easy"
    ),
    ModelTier(
        tier_name="Security-Specialized",
        models=[
            "PRIMUS",
            "RedSage",
            "Foundation-Sec-8B",
        ],
        expected_difficulty="Very Easy"
    ),
]

SECURE_TASKS = ["MAET", "CWET", "KCV"]


def run_model_evaluation(
    model_name: str,
    is_base: bool = True
) -> Dict[str, float]:
    """
    Run evaluation for a model on SECURE tasks.

    Returns: {task_name: accuracy}
    """

    results = {}

    for task in SECURE_TASKS:
        logger.info(f"Evaluating {model_name} on SECURE/{task}...")

        # Sanitize model name for file paths (replace "/" with "_")
        safe_model_name = model_name.replace("/", "_")

        cmd = [
            "python",
            "../run_evaluate_llm_judge.py",
            "--model_path", model_name,
            "--tasks", task,
            "--max_samples", "50",  # Reasonable sample size for ceiling detection
            "--output", f"results/temp_{safe_model_name}_{task}"
        ]

        if is_base:
            cmd.append("--is_base")

        try:
            subprocess.run(cmd, check=True, capture_output=True)

            # Parse result - judge outputs to {output}_detailed/results.json
            detailed_dir = f"results/temp_{safe_model_name}_{task}_detailed"
            result_file = Path(detailed_dir) / "results.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    # Judge output format: {"tasks": {task_name: {accuracy, correct, total}}}
                    tasks_data = data.get("tasks", {})
                    if task in tasks_data:
                        accuracy = tasks_data[task].get("accuracy", 0.0)
                        results[task] = accuracy
                    else:
                        logger.warning(f"Task {task} not found in results for {model_name}")
                        results[task] = None

                # Clean up temp directory
                import shutil
                shutil.rmtree(detailed_dir, ignore_errors=True)
            else:
                logger.warning(f"No result file for {model_name}/{task}")
                results[task] = None

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to evaluate {model_name}/{task}: {e}")
            results[task] = None

    return results


def analyze_ceiling_evidence(all_results: Dict) -> Dict:
    """Analyze whether ceiling effects exist"""

    analysis = {
        "ceiling_threshold": 0.90,  # 90% = potential ceiling
        "tasks_with_ceiling": [],
        "model_tiers_summary": {},
        "interpretation": ""
    }

    for model_name in all_results:
        results = all_results[model_name]
        for task, accuracy in results.items():
            if accuracy is not None and accuracy >= analysis["ceiling_threshold"]:
                analysis["tasks_with_ceiling"].append({
                    "task": task,
                    "model": model_name,
                    "accuracy": accuracy
                })

    # Summarize by tier
    for tier in BASELINE_TIERS:
        tier_accuracies = []
        for model in tier.models:
            if model in all_results:
                for task, acc in all_results[model].items():
                    if acc is not None:
                        tier_accuracies.append(acc)

        if tier_accuracies:
            analysis["model_tiers_summary"][tier.tier_name] = {
                "mean_accuracy": float(np.mean(tier_accuracies)),
                "std_accuracy": float(np.std(tier_accuracies)),
                "num_models": len(tier.models),
                "expected_difficulty": tier.expected_difficulty
            }

    # Interpretation
    if len(analysis["tasks_with_ceiling"]) >= 3:  # All 3 SECURE tasks
        analysis["interpretation"] = (
            "CRITICAL: All SECURE tasks show ceiling effects. "
            "These tasks provide zero discrimination power and should be excluded "
            "or replaced with harder variants."
        )
    elif len(analysis["tasks_with_ceiling"]) > 0:
        analysis["interpretation"] = (
            "MODERATE: Some SECURE tasks show ceiling effects. "
            "Consider reviewing task difficulty and validation methodology."
        )
    else:
        analysis["interpretation"] = (
            "GOOD: No ceiling effects detected. Tasks discriminate across model tiers."
        )

    return analysis


def generate_csv_summary(all_results: Dict, output_file: str) -> None:
    """Generate CSV for easy comparison"""

    all_models = sorted(all_results.keys())

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Model"] + SECURE_TASKS
        writer.writerow(header)

        for model in all_models:
            row = [model]
            for task in SECURE_TASKS:
                acc = all_results[model].get(task)
                if acc is not None:
                    row.append(f"{acc:.3f}")
                else:
                    row.append("N/A")
            writer.writerow(row)

    logger.info(f"✓ CSV summary saved to {output_file}")


def generate_markdown_report(analysis: Dict, output_file: str) -> None:
    """Generate markdown report"""

    report = []
    report.append("# SECURE Benchmark Ceiling Effect Analysis")
    report.append("")
    report.append("## Methodology")
    report.append(
        "We test multiple baseline models of varying capability on SECURE tasks "
        "to detect ceiling effects (tasks where performance saturates across all models)."
    )
    report.append("")
    report.append(f"Ceiling threshold: {analysis['ceiling_threshold']:.0%}")
    report.append("")

    report.append("## Findings")
    report.append(f"Tasks with ceiling effects: {len(analysis['tasks_with_ceiling'])}")
    report.append("")

    if analysis["tasks_with_ceiling"]:
        report.append("### Affected Tasks")
        for item in analysis["tasks_with_ceiling"]:
            report.append(
                f"- **{item['task']}** (Model: {item['model']}, Accuracy: {item['accuracy']:.3f})"
            )
        report.append("")

    report.append("## Model Tier Summary")
    report.append("")
    report.append("| Tier | Mean Accuracy | Std Dev | Models | Expected Difficulty |")
    report.append("|------|---------------|---------|--------|---------------------|")
    for tier_name, stats in analysis["model_tiers_summary"].items():
        report.append(
            f"| {tier_name} | {stats['mean_accuracy']:.3f} | {stats['std_accuracy']:.3f} | "
            f"{stats['num_models']} | {stats['expected_difficulty']} |"
        )
    report.append("")

    report.append("## Interpretation")
    report.append(f"> {analysis['interpretation']}")
    report.append("")

    report.append("## Recommendations")
    if "CRITICAL" in analysis["interpretation"]:
        report.append(
            "1. **Exclude ceiling tasks** from model comparisons until they are replaced\n"
            "2. **Increase task difficulty** by selecting harder ICS vulnerabilities\n"
            "3. **Add open-ended questions** instead of 4-choice MCQ format"
        )
    else:
        report.append(
            "1. **Monitor task difficulty** over time\n"
            "2. **Track model performance improvements** to detect future saturation"
        )

    with open(output_file, 'w') as f:
        f.write('\n'.join(report))

    logger.info(f"✓ Report saved to {output_file}")


def main():
    logger.info("=" * 70)
    logger.info("SECURE BENCHMARK CEILING EFFECT ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Testing {sum(len(tier.models) for tier in BASELINE_TIERS)} models across {len(SECURE_TASKS)} tasks")
    logger.info("")

    all_results = {}

    # Loop through model tiers
    for tier in BASELINE_TIERS:
        logger.info(f"[Tier] {tier.tier_name}")

        for model_name in tier.models:
            logger.info(f"  Testing {model_name}...")
            results = run_model_evaluation(model_name, is_base=True)
            all_results[model_name] = results

            # Log results for this model
            for task, acc in results.items():
                if acc is not None:
                    logger.info(f"    {task}: {acc:.3f}")
                else:
                    logger.warning(f"    {task}: FAILED")

    # Analyze
    logger.info("")
    logger.info("Analyzing ceiling effects...")
    analysis = analyze_ceiling_evidence(all_results)

    # Save results
    result_data = {
        "models_tested": list(all_results.keys()),
        "tasks_tested": SECURE_TASKS,
        "analysis": analysis
    }

    with open(OUTPUT_DIR / "secure_ceiling_analysis.json", 'w') as f:
        json.dump(result_data, f, indent=2)
    logger.info(f"✓ Analysis saved to {OUTPUT_DIR}/secure_ceiling_analysis.json")

    # Generate CSV
    generate_csv_summary(all_results, OUTPUT_DIR / "secure_model_performance.csv")

    # Generate report
    generate_markdown_report(analysis, OUTPUT_DIR / "secure_ceiling_evidence.md")

    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ SECURE Ceiling Analysis complete!")
    logger.info(analysis["interpretation"])
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
