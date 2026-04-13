#!/usr/bin/env python3
"""
Generate framework evaluation heatmap and visualization.

Creates visual representations of Table 2 from the paper:
- Framework scoring matrix (D1-D6 across 11 benchmarks)
- Shows which benchmarks score H/M/L/N on each dimension

Outputs:
- results/framework_heatmap.pdf
- results/framework_heatmap.png
- results/framework_interactive.html
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# Framework scoring data (from paper Table 2)
BENCHMARKS = [
    "SecEval",
    "CyberMetric",
    "CTI-Bench*",
    "SECURE*",
    "SEvenLLM-Bench",
    "AthenaBench*",
    "SecBench",
    "CISSP",
    "CTIARENA",
    "RedSage-Bench",
    "MMLU-CS"
]

DIMENSIONS = ["D1: Scale", "D2: Validation", "D3: Diversity", "D4: Eval.Robust", "D5: Repro", "D6: Multi"]

# Scoring: H=3, M=2, L=1, N=0
SCORES = {
    "SecEval":        [2, 1, 2, 1, 2, 0],
    "CyberMetric":    [2, 3, 1, 2, 3, 0],
    "CTI-Bench":      [2, 0, 3, 1, 2, 0],
    "SECURE":         [0, 0, 2, 2, 1, 0],
    "SEvenLLM-Bench": [2, 0, 2, 2, 0, 2],
    "AthenaBench":    [3, 0, 3, 1, 2, 0],
    "SecBench":       [3, 3, 2, 2, 2, 2],
    "CISSP":          [2, 1, 1, 3, 0, 0],
    "CTIARENA":       [2, 3, 3, 3, 0, 0],
    "RedSage-Bench":  [3, 1, 2, 2, 2, 0],
    "MMLU-CS":        [1, 2, 1, 3, 3, 0]
}

# Dimension descriptions for legend
DIMENSION_DESCRIPTIONS = {
    "D1: Scale": "500+ items per task",
    "D2: Validation": "Expert/contest-sourced",
    "D3: Diversity": "3+ task paradigms",
    "D4: Eval.Robust": "LLM-judge for extraction",
    "D5: Repro": "Code + data released",
    "D6: Multi": "3+ language families"
}


def create_heatmap(output_file: str, format: str = "png") -> None:
    """Generate heatmap visualization"""

    # Prepare data matrix
    benchmark_list = [b.replace("*", "") for b in BENCHMARKS]
    scores_matrix = np.array([SCORES[b] for b in benchmark_list])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(scores_matrix.T, cmap="RdYlGn", aspect="auto", vmin=0, vmax=3)

    # Set labels
    ax.set_xticks(np.arange(len(benchmark_list)))
    ax.set_yticks(np.arange(len(DIMENSIONS)))
    ax.set_xticklabels(benchmark_list, rotation=45, ha="right")
    ax.set_yticklabels(DIMENSIONS)

    # Add text annotations
    for i in range(len(benchmark_list)):
        for j in range(len(DIMENSIONS)):
            score = scores_matrix[i, j]
            if score == 0:
                text_val = "✗"
                color = "white"
            elif score == 1:
                text_val = "L"
                color = "white"
            elif score == 2:
                text_val = "M"
                color = "black"
            else:  # score == 3
                text_val = "H"
                color = "black"

            ax.text(i, j, text_val, ha="center", va="center", color=color, fontweight="bold", fontsize=12)

    # Title and labels
    ax.set_title("Cybersecurity LLM Benchmark Framework Evaluation\n(H=High, M=Medium, L=Low, ✗=None/Undisclosed)",
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Benchmark Family", fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluation Dimension", fontsize=12, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Coverage", fontsize=11)

    # Add RISys cluster marker
    ax.text(2, -1.5, "* = RISys-Lab cluster", fontsize=10, style="italic")

    plt.tight_layout()

    # Save
    save_path = OUTPUT_DIR / f"framework_heatmap.{format}"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Heatmap saved to {save_path}")

    plt.close()


def create_scatter_plot(output_file: str = "framework_scatter.png") -> None:
    """Create scatter plot: # items vs validation provenance"""

    benchmark_list = [b.replace("*", "") for b in BENCHMARKS]

    # Items per benchmark (approximate from paper)
    items = {
        "SecEval": 2189,
        "CyberMetric": 500,
        "CTI-Bench": 4610,
        "SECURE": 0,  # Unknown
        "SEvenLLM-Bench": 1300,
        "AthenaBench": 8100,
        "SecBench": 47910,
        "CISSP": 3015,
        "CTIARENA": 691,
        "RedSage-Bench": 30240,
        "MMLU-CS": 116
    }

    # Validation scores (D2)
    validation = [SCORES[b][1] for b in benchmark_list]
    item_counts = [items[b] for b in benchmark_list]
    benchmark_list_clean = [b.replace("*", "") for b in BENCHMARKS]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Create scatter
    scatter = ax.scatter(item_counts, validation, s=200, alpha=0.6, c=np.arange(len(benchmark_list_clean)),
                        cmap="tab20", edgecolors="black", linewidth=1.5)

    # Add labels
    for i, label in enumerate(benchmark_list_clean):
        ax.annotate(label, (item_counts[i], validation[i]), fontsize=9, ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Benchmark Size (# items, log scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation Provenance (D2 Score)", fontsize=12, fontweight="bold")
    ax.set_title("Benchmark Scale vs. Validation Quality", fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 3.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_file, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Scatter plot saved to {OUTPUT_DIR}/{output_file}")
    plt.close()


def create_dimension_summary_bar(output_file: str = "framework_dimensions.png") -> None:
    """Create bar chart showing dimension-wise coverage"""

    benchmark_list = [b.replace("*", "") for b in BENCHMARKS]

    # Count H/M/L/N for each dimension
    all_scores = np.array([SCORES[b] for b in benchmark_list])
    dimension_stats = []

    for dim_idx in range(len(DIMENSIONS)):
        scores = all_scores[:, dim_idx]
        high_count = np.sum(scores == 3)
        medium_count = np.sum(scores == 2)
        low_count = np.sum(scores == 1)
        none_count = np.sum(scores == 0)

        dimension_stats.append({
            "dimension": DIMENSIONS[dim_idx],
            "High": high_count,
            "Medium": medium_count,
            "Low": low_count,
            "None": none_count
        })

    # Create stacked bar
    fig, ax = plt.subplots(figsize=(12, 6))

    dimensions = [s["dimension"] for s in dimension_stats]
    high = [s["High"] for s in dimension_stats]
    medium = [s["Medium"] for s in dimension_stats]
    low = [s["Low"] for s in dimension_stats]
    none = [s["None"] for s in dimension_stats]

    x = np.arange(len(dimensions))
    width = 0.6

    ax.bar(x, high, width, label="High (H)", color="#2ecc71")
    ax.bar(x, medium, width, bottom=high, label="Medium (M)", color="#f39c12")
    ax.bar(x, low, width, bottom=np.array(high) + np.array(medium), label="Low (L)", color="#e74c3c")
    ax.bar(x, none, width, bottom=np.array(high) + np.array(medium) + np.array(low),
           label="None (✗)", color="#95a5a6")

    ax.set_ylabel("Number of Benchmarks", fontsize=12, fontweight="bold")
    ax.set_title("Framework Dimension Coverage Across All Benchmarks", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, len(benchmark_list) + 1)

    # Add value labels
    for i, dim in enumerate(dimensions):
        total = len(benchmark_list)
        ax.text(i, total + 0.2, f"{len(benchmark_list)}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_file, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Dimension summary saved to {OUTPUT_DIR}/{output_file}")
    plt.close()


def create_html_interactive(output_file: str = "framework_interactive.html") -> None:
    """Create interactive HTML visualization using plotly"""

    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        logger.warning("Plotly not installed, skipping interactive visualization")
        return

    benchmark_list_clean = [b.replace("*", "") for b in BENCHMARKS]
    all_scores = np.array([SCORES[b] for b in benchmark_list_clean])

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=all_scores.T,
        x=benchmark_list_clean,
        y=DIMENSIONS,
        colorscale="RdYlGn",
        zmin=0,
        zmax=3,
        text=[["H" if v == 3 else "M" if v == 2 else "L" if v == 1 else "✗"
               for v in row] for row in all_scores.T],
        texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate="<b>%{x}</b><br>%{y}<br>Score: %{z}<extra></extra>",
        colorbar=dict(title="Coverage Level")
    ))

    fig.update_layout(
        title="Cybersecurity LLM Benchmark Framework Evaluation (Interactive)",
        xaxis_title="Benchmark Family",
        yaxis_title="Evaluation Dimension",
        width=1200,
        height=500,
        font=dict(size=12)
    )

    fig.write_html(OUTPUT_DIR / output_file)
    logger.info(f"✓ Interactive visualization saved to {OUTPUT_DIR}/{output_file}")


def create_summary_json(output_file: str = "framework_summary.json") -> None:
    """Save framework data as structured JSON"""

    benchmark_list = [b.replace("*", "") for b in BENCHMARKS]

    summary = {
        "metadata": {
            "title": "Cybersecurity LLM Benchmark Framework Evaluation",
            "benchmarks_evaluated": len(BENCHMARKS),
            "dimensions": len(DIMENSIONS)
        },
        "dimensions": {
            dim: {"description": DIMENSION_DESCRIPTIONS.get(dim, "")}
            for dim in DIMENSIONS
        },
        "benchmarks": {},
        "dimension_coverage": {}
    }

    # Aggregate scores by dimension
    all_scores_array = np.array([SCORES[b] for b in benchmark_list])
    for dim_idx, dim in enumerate(DIMENSIONS):
        scores = all_scores_array[:, dim_idx]
        summary["dimension_coverage"][dim] = {
            "high": int(np.sum(scores == 3)),
            "medium": int(np.sum(scores == 2)),
            "low": int(np.sum(scores == 1)),
            "none": int(np.sum(scores == 0))
        }

    # Per-benchmark scores
    for bench_idx, benchmark in enumerate(benchmark_list):
        summary["benchmarks"][benchmark] = {
            "scores": {DIMENSIONS[i]: SCORES[benchmark][i] for i in range(len(DIMENSIONS))},
            "total_score": sum(SCORES[benchmark]),
            "is_risys_cluster": benchmark in ["CTI-Bench", "SECURE", "AthenaBench"]
        }

    with open(OUTPUT_DIR / output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✓ Summary saved to {OUTPUT_DIR}/{output_file}")


def main():
    benchmark_list = [b.replace("*", "") for b in BENCHMARKS]

    logger.info("=" * 70)
    logger.info("GENERATING FRAMEWORK VISUALIZATIONS")
    logger.info("=" * 70)

    logger.info("Creating heatmap...")
    create_heatmap(OUTPUT_DIR / "framework_heatmap.png", format="png")

    logger.info("Creating scatter plot...")
    create_scatter_plot()

    logger.info("Creating dimension summary...")
    create_dimension_summary_bar()

    logger.info("Creating interactive HTML...")
    create_html_interactive()

    logger.info("Saving summary JSON...")
    create_summary_json()

    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ All visualizations complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
