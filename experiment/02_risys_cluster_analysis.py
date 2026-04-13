#!/usr/bin/env python3
"""
Analyze RISys-Lab benchmark cluster circularity.

Hypothesis: RedSage (RISys-Lab model) scores significantly higher on
RISys-Lab benchmarks (CTI-Bench, AthenaBench, SECURE) vs. independent
benchmarks (CyberMetric, SecEval, CISSP).

This script compares performance across benchmark categories to detect
potential circular evaluation patterns.

Output:
- results/risys_cluster_analysis.json
- results/risys_circular_evidence.md (report)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


@dataclass
class BenchmarkCluster:
    """Categorize benchmarks by author affiliation"""
    name: str
    benchmarks: List[str]
    is_risys: bool


# Define benchmark clusters
CLUSTERS = [
    BenchmarkCluster(
        name="RISys-Lab Benchmarks",
        benchmarks=["CTI-Bench", "AthenaBench", "SECURE"],
        is_risys=True
    ),
    BenchmarkCluster(
        name="Independent Benchmarks",
        benchmarks=["CyberMetric", "SecEval", "CISSP", "MMLU-CS", "SecBench"],
        is_risys=False
    ),
]


@dataclass
class ModelPerformance:
    """Track model performance across benchmark categories"""
    model_name: str
    risys_avg: float = 0.0
    risys_std: float = 0.0
    independent_avg: float = 0.0
    independent_std: float = 0.0
    difference: float = 0.0  # risys - independent
    difference_ratio: float = 0.0  # (risys - independent) / independent

    def __str__(self):
        return (f"{self.model_name}: "
                f"RISys={self.risys_avg:.3f}, Independent={self.independent_avg:.3f}, "
                f"Diff={self.difference:.3f} ({self.difference_ratio:.1%})")


def load_evaluation_results(results_file: str) -> Dict:
    """Load full-scale evaluation results"""
    with open(results_file, 'r') as f:
        return json.load(f)


def extract_model_cluster_performance(
    results: Dict,
    model_name: str,
    clusters: List[BenchmarkCluster]
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Extract model accuracy scores by benchmark cluster"""

    cluster_scores = {}
    for cluster in clusters:
        cluster_scores[cluster.name] = []

    # Iterate through results
    for result in results.get("results", []):
        if result["model_name"] != model_name:
            continue

        benchmark = result["benchmark"]
        accuracy = result["accuracy"]

        # Find which cluster this benchmark belongs to
        for cluster in clusters:
            if benchmark in cluster.benchmarks:
                cluster_scores[cluster.name].append(accuracy)
                break

    return cluster_scores


def compute_cluster_statistics(
    all_results: Dict,
    clusters: List[BenchmarkCluster]
) -> List[ModelPerformance]:
    """Compute average performance for each model on each cluster"""

    models = set(r["model_name"] for r in all_results.get("results", []))
    performances = []

    for model_name in sorted(models):
        cluster_scores = extract_model_cluster_performance(all_results, model_name, clusters)

        risys_scores = cluster_scores.get("RISys-Lab Benchmarks", [])
        independent_scores = cluster_scores.get("Independent Benchmarks", [])

        perf = ModelPerformance(model_name=model_name)

        if risys_scores:
            perf.risys_avg = np.mean(risys_scores)
            perf.risys_std = np.std(risys_scores)

        if independent_scores:
            perf.independent_avg = np.mean(independent_scores)
            perf.independent_std = np.std(independent_scores)

        if perf.independent_avg > 0:
            perf.difference = perf.risys_avg - perf.independent_avg
            perf.difference_ratio = perf.difference / perf.independent_avg

        performances.append(perf)

    return performances


def analyze_circular_evidence(
    performances: List[ModelPerformance],
    critical_threshold: float = 0.15
) -> Dict:
    """Analyze whether RedSage shows circular evaluation pattern"""

    analysis = {
        "critical_threshold": f"{critical_threshold:.1%}",
        "red_flags": [],
        "observations": [],
        "models_analyzed": []
    }

    for perf in performances:
        analysis["models_analyzed"].append({
            "model": perf.model_name,
            "risys_avg": round(perf.risys_avg, 4),
            "independent_avg": round(perf.independent_avg, 4),
            "difference": round(perf.difference, 4),
            "difference_ratio": round(perf.difference_ratio, 4)
        })

        # RedSage-specific analysis
        if "RedSage" in perf.model_name or "redsage" in perf.model_name.lower():
            if perf.difference_ratio > critical_threshold:
                analysis["red_flags"].append(
                    f"[CIRCULAR EVIDENCE] {perf.model_name}: "
                    f"{perf.difference_ratio:.1%} higher on OWN benchmarks "
                    f"({perf.risys_avg:.3f} vs {perf.independent_avg:.3f})"
                )
            else:
                analysis["observations"].append(
                    f"{perf.model_name}: Moderate benchmark-specific advantage "
                    f"({perf.difference_ratio:.1%})"
                )

        # General observation: security-specialized models should improve equally
        if perf.model_name != "Llama-3.1-8B-Instruct":  # Not base model
            if perf.risys_avg > perf.independent_avg:
                analysis["observations"].append(
                    f"{perf.model_name}: Stronger on RISys benchmarks "
                    f"(+{perf.difference_ratio:.1%})"
                )

    # Comparative analysis
    redsage = next((p for p in performances if "RedSage" in p.model_name), None)
    primus = next((p for p in performances if "PRIMUS" in p.model_name), None)

    if redsage and primus:
        redsage_to_primus_risys = redsage.risys_avg - primus.risys_avg
        redsage_to_primus_indep = redsage.independent_avg - primus.independent_avg

        analysis["comparative_analysis"] = {
            "redsage_vs_primus_on_risys": round(redsage_to_primus_risys, 4),
            "redsage_vs_primus_on_independent": round(redsage_to_primus_indep, 4),
            "advantage_concentration": {
                "risys_specific": redsage_to_primus_risys > redsage_to_primus_indep,
                "difference": round(redsage_to_primus_risys - redsage_to_primus_indep, 4),
                "interpretation": (
                    "RedSage shows greater advantage on its own benchmarks, "
                    "suggesting potential benchmark-specific tuning"
                    if redsage_to_primus_risys > redsage_to_primus_indep else
                    "Performance advantage distributed across benchmark types"
                )
            }
        }

    return analysis


def generate_markdown_report(
    analysis: Dict,
    performances: List[ModelPerformance],
    output_file: str
) -> None:
    """Generate human-readable markdown report"""

    report = []
    report.append("# RISys-Lab Benchmark Cluster Analysis")
    report.append("")
    report.append("## Methodology")
    report.append(
        "This analysis investigates whether RedSage (RISys-Lab model) shows "
        "disproportionately higher performance on benchmarks from the same "
        "research group (CTI-Bench, AthenaBench, SECURE) compared to independent "
        "benchmarks (CyberMetric, SecEval, CISSP, MMLU-CS, SecBench)."
    )
    report.append("")

    report.append("## Critical Findings")
    if analysis["red_flags"]:
        for flag in analysis["red_flags"]:
            report.append(f"⚠️  {flag}")
    else:
        report.append("✓ No critical circular evaluation evidence detected")
    report.append("")

    report.append("## Performance Comparison by Model")
    report.append("")
    report.append("| Model | RISys Avg | Independent Avg | Diff | Diff % |")
    report.append("|-------|-----------|-----------------|------|--------|")
    for model_data in analysis["models_analyzed"]:
        report.append(
            f"| {model_data['model']} | "
            f"{model_data['risys_avg']:.3f} | "
            f"{model_data['independent_avg']:.3f} | "
            f"{model_data['difference']:+.3f} | "
            f"{model_data['difference_ratio']:+.1%} |"
        )
    report.append("")

    report.append("## Observations")
    for obs in analysis["observations"]:
        report.append(f"- {obs}")
    report.append("")

    if "comparative_analysis" in analysis:
        report.append("## Comparative Analysis: RedSage vs PRIMUS")
        comp = analysis["comparative_analysis"]
        report.append(f"- RedSage advantage on RISys benchmarks: {comp['redsage_vs_primus_on_risys']:+.3f}")
        report.append(f"- RedSage advantage on independent benchmarks: {comp['redsage_vs_primus_on_independent']:+.3f}")
        report.append(f"- Advantage concentration on RISys: {comp['risys_specific']}")
        report.append(f"- Interpretation: {comp['interpretation']}")
    report.append("")

    report.append("## Interpretation")
    report.append(f"Critical threshold: {analysis['critical_threshold']}")
    report.append(
        "If a model shows >15% higher performance on its author group's benchmarks, "
        "this suggests potential benchmark-specific tuning or circular evaluation."
    )

    with open(output_file, 'w') as f:
        f.write('\n'.join(report))

    logger.info(f"✓ Report saved to {output_file}")


def main():
    import sys

    results_file = OUTPUT_DIR / "full_scale_results.json"

    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.info("Run 01_full_scale_evaluation.py first to generate results")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("RISYS-LAB CLUSTER CIRCULARITY ANALYSIS")
    logger.info("=" * 70)

    # Load results
    logger.info("Loading evaluation results...")
    results = load_evaluation_results(str(results_file))

    # Compute statistics
    logger.info("Computing cluster statistics...")
    performances = compute_cluster_statistics(results, CLUSTERS)

    for perf in performances:
        logger.info(str(perf))

    # Analyze circular evidence
    logger.info("Analyzing circular evaluation patterns...")
    analysis = analyze_circular_evidence(performances)

    # Save analysis
    with open(OUTPUT_DIR / "risys_cluster_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"✓ Analysis saved to {OUTPUT_DIR}/risys_cluster_analysis.json")

    # Generate markdown report
    generate_markdown_report(analysis, performances, OUTPUT_DIR / "risys_circular_evidence.md")

    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ Analysis complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
