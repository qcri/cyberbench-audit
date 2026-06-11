"""Generate the analysis walkthrough notebooks (.ipynb) from the cell content
defined here. This file is the editable source of truth; run it to (re)emit the
notebooks:  python build.py

Notebooks are thin: they import the existing ``analysis`` modules (via
``nbtools.run_mod``, which imports inside a try/except so optional-dep import
failures never crash a notebook) and render cached ``reports/`` artifacts. No
analysis logic is duplicated.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SETUP = """\
import nbtools as nb
from nbtools import show_df, show_fig, show_md, run_mod, heavy, REGEN, REPORTS_DIR
print("SAYF_NB_REGEN =", REGEN, "  (heavy GPU/API steps run only when True)")
print("reports dir  :", REPORTS_DIR)\
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def setup_cell():
    return code(SETUP)


# ───────────────────────────── 00 — overview ─────────────────────────────
NB00 = [
    md("""\
# Sayf-Eval analysis walkthrough

A guided, runnable tour of the analysis behind *"Benchmark Scores Are
Pipeline-Dependent: A Reliability Audit of Cybersecurity LLM Benchmarks."*

Each notebook **reuses** the `analysis/` modules (no logic is copied) and renders
the already-computed artifacts in `analysis/reports/`. Light steps recompute
live; heavy steps (GPU/API) are rendered from cache unless you opt in.

## How to use
- Run cells top-to-bottom. Light steps recompute in seconds–minutes; if
  `outputs/` isn't mounted they fall back to the cached `reports/` artifact.
- **Heavy steps are off by default.** To recompute embeddings (GPU) or the
  verification / K-A judges (Azure API), launch Jupyter with `SAYF_NB_REGEN=1`.
- Kernel: any Python ≥3.10 with the analysis deps + `requirements-notebooks.txt`.

## The measurement-pipeline framing
Every benchmark is modeled as a 5-stage pipeline — **Dataset 𝒟 · Prompt 𝒫 ·
Inference 𝓘 · Extraction/scoring 𝓔 · Aggregation 𝒜** — and a reported score is
conditional on the whole pipeline, not an intrinsic model property. The analysis
quantifies 15 recurring failure modes across these stages.

## Notebook map
| Notebook | Theme | Failure modes |
|---|---|---|
| `01_results_table` | Master accuracy table + secondary metrics (F1, MAD) | 𝒜 aggregation |
| `02_judge_agreement` | LLM-judge reliability (Cohen's κ across judges) | 𝓔 extraction/scoring |
| `03_gold_errors_verification` | Suspect gold labels + search-grounded verification | 𝐹₂(𝒟) label quality |
| `04_capability_coverage` | Knowledge-vs-Analytical coverage | 𝐹₁(𝒟) limited coverage |
| `05_redundancy_correlation_embeddings` | Cross-task redundancy, effective dimensions | benchmark redundancy |
"""),
    setup_cell(),
    md("### What's already computed (`reports/` inventory)"),
    code("""\
for p in sorted(REPORTS_DIR.iterdir()):
    print(("dir  " if p.is_dir() else "file ") + p.name)\
"""),
    md("""\
### Run order
`results_table` produces `per_model_task_accuracy.json`, consumed by
`gold_error_voting` and `correlation`. Otherwise the notebooks are independent;
`00`→`05` is a sensible reading order.
"""),
]

# ─────────────────────────── 01 — results table ──────────────────────────
NB01 = [
    md("""\
# 01 · Master results table

The headline table: per-(model, task) accuracy as a **per-sample majority vote**
across the available judge runs, then `accuracy = mean(majority_correct)`. This
addresses aggregation failure modes — denominator = all attempted items, with a
single documented scoring convention.
"""),
    setup_cell(),
    md("### Accuracy table — `analysis.results_table`\nMajority-vote accuracy over judges (`cell_accuracy`)."),
    code("""\
run_mod("results_table", "analysis.results_table")
df = show_df("results_table.csv")\
"""),
    md("### Secondary metrics — `analysis.secondary_metrics`\nMicro-F1 for ID-extraction tasks (ATE), MAD for CVSS vectors (VSP)."),
    code("""\
run_mod("secondary_metrics", "analysis.secondary_metrics")
show_df("secondary_metrics.csv")\
"""),
    md("### Accuracy heatmap — `analysis.make_plots`"),
    code("""\
run_mod("make_plots", "analysis.make_plots")
show_fig("accuracy_heatmap.png")
show_fig("per_model_average.png")\
"""),
    md("### LaTeX table — `analysis.build_results_latex`\nThe `tab:main_results` block used in the paper."),
    code('run_mod("build_results_latex", "analysis.build_results_latex")'),
    md("_Appears in the paper as the main results table (Section 5)._"),
]

# ───────────────────────── 02 — judge agreement ──────────────────────────
NB02 = [
    md("""\
# 02 · LLM-judge reliability

The unified judge does extraction + verdict. How reliable is it? We compare
independent judge runs per sample — Cohen's κ and raw agreement — which probes
the extraction/scoring stage (𝓔).
"""),
    setup_cell(),
    md("### 3-judge agreement — `analysis.judge_agreement`"),
    code("""\
run_mod("judge_agreement", "analysis.judge_agreement")
show_df("judge_agreement/per_cell_default_vs_v1.csv")\
"""),
    md("### Summary (κ averages, per-task/per-model, lowest-κ cells)"),
    code('show_md("judge_agreement/summary.md")'),
    code('show_fig("agent_agreement.png")'),
    md("_Supports the judge-reliability discussion ($\\\\mathcal{F}(\\\\mathcal{E})$)._"),
]

# ──────────────────── 03 — gold errors + verification ────────────────────
NB03 = [
    md("""\
# 03 · Gold-label correctness ($\\mathcal{F}_2(\\mathcal{D})$)

Are the benchmark *answers* right? We flag suspect gold labels by majority vote
across model predictions, then audit a sample with **search-grounded** and
**direct** GPT-5.4 verifiers, and measure the impact of removing confirmed
mislabels on the rankings.
"""),
    setup_cell(),
    md("### Flag suspect labels — `analysis.gold_error_voting`\nThreshold sweep + weighted / top-k / acceptance voting."),
    code("""\
run_mod("gold_error_voting", "analysis.gold_error_voting")
show_df("gold_errors/threshold_sweep.csv")
show_fig("threshold_sweep.png")\
"""),
    md("### Build the verification bank — `analysis.build_verification_bank`"),
    code('run_mod("build_verification_bank", "analysis.build_verification_bank")'),
    md("""\
### Verify flagged labels — `analysis.verify`  *(heavy · Azure API)*
Off by default — rendered from cached verdicts. Set `SAYF_NB_REGEN=1` (and an
Azure key) to re-run the search + direct verifier agents.
"""),
    code("""\
if heavy("verify"):
    run_mod("verify", "analysis.verify")
show_df("verification/per_threshold_search.csv")
show_md("verification/summary.md")\
"""),
    md("### Aggregate verdicts + figures — `aggregate_verification`, `make_verify_plots`, `make_fp_threshold_plot`"),
    code("""\
run_mod("aggregate_verification", "analysis.aggregate_verification")
run_mod("make_verify_plots", "analysis.make_verify_plots")
run_mod("make_fp_threshold_plot", "analysis.make_fp_threshold_plot")
show_fig("fp_vs_threshold.png")
show_fig("fp_threshold_combined.png")
show_fig("verdict_breakdown.png")\
"""),
    md("### Impact on rankings — `analysis.label_quality_impact`\nRecompute accuracy excluding confirmed-mislabel samples; compare rankings."),
    code("""\
run_mod("label_quality_impact", "analysis.label_quality_impact")
show_df("label_quality_impact/delta_table.csv")
show_md("label_quality_impact/ranking_diff.md")\
"""),
    md("_Confirmed-label-error rate and the post-mitigation rank changes are the $\\\\mathcal{F}_2(\\\\mathcal{D})$ evidence._"),
]

# ───────────────────── 04 — capability coverage (K/A) ─────────────────────
NB04 = [
    md("""\
# 04 · Capability coverage — Knowledge vs Analytical ($\\mathcal{F}_1(\\mathcal{D})$)

Do these benchmarks test reasoning, or mostly factual recall? We classify a
stratified sample of every task's questions as **Knowledge** or **Analytical**
(multi-model vote) and break it down per task/parent.
"""),
    setup_cell(),
    md("### Build the K/A sample bank — `analysis.build_ka_bank`"),
    code('run_mod("build_ka_bank", "analysis.build_ka_bank")'),
    md("""\
### Classify K vs A — `analysis.classify_ka`  *(heavy · Azure/vLLM)*
Off by default — rendered from cached verdicts. `SAYF_NB_REGEN=1` to re-run.
"""),
    code("""\
if heavy("classify_ka"):
    run_mod("classify_ka", "analysis.classify_ka")\
"""),
    md("### Aggregate — `analysis.aggregate_ka`\nMajority K/A + inter-rater agreement (Cohen's / Fleiss' κ)."),
    code("""\
run_mod("aggregate_ka", "analysis.aggregate_ka")
show_df("coverage/per_parent_breakdown.csv")
show_df("coverage/per_task_breakdown.csv")
show_md("coverage/summary.md")\
"""),
    md("### Coverage figures — `analysis.make_coverage_plots`"),
    code("""\
run_mod("make_coverage_plots", "analysis.make_coverage_plots")
show_fig("coverage_main.png")
show_fig("coverage_appendix.png")\
"""),
    md("_Shows the fraction of knowledge-oriented items per benchmark ($\\\\mathcal{F}_1(\\\\mathcal{D})$)._"),
]

# ───────────── 05 — redundancy: correlation + embeddings ──────────────────
NB05 = [
    md("""\
# 05 · Cross-task redundancy & effective dimensions

How many *independent* things does the suite measure? We correlate the 24×N
accuracy matrix (Kendall τ), estimate effective dimensionality (PCA + Horn's
parallel analysis), and corroborate with **question-embedding** similarity.
"""),
    setup_cell(),
    md("### Correlation structure — `analysis.correlation`\nKendall τ / Spearman / Pearson, bootstrap CI, PCA, clustering."),
    code("""\
run_mod("correlation", "analysis.correlation")
show_df("correlation/pairwise_kendall.csv")\
"""),
    md("### Effective dimensions + redundant pairs"),
    code("""\
import json
eff = json.loads((REPORTS_DIR / "correlation/effective_dimensions.json").read_text())
print(json.dumps(eff, indent=2))
show_md("correlation/redundant_pairs.md")\
"""),
    md("### Correlation figures — `analysis.make_corr_plots`"),
    code("""\
run_mod("make_corr_plots", "analysis.make_corr_plots")
show_fig("kendall_heatmap_clustered.png")
show_fig("pca_scree.png")
show_fig("dendrogram.png")\
"""),
    md("""\
### Question embeddings — `analysis.embed`  *(heavy · GPU)*
Off by default — rendered from cached `embeddings/`. `SAYF_NB_REGEN=1` (GPU) to
recompute sentence-transformer embeddings.
"""),
    code("""\
if heavy("embed"):
    run_mod("embed", "analysis.embed")\
"""),
    md("### Semantic redundancy — `embedding_correlation` + `make_embed_plots`\nCentroid similarity, EVoC clustering, Mantel test vs Kendall τ."),
    code("""\
run_mod("embedding_correlation", "analysis.embedding_correlation")
run_mod("make_embed_plots", "analysis.make_embed_plots")
show_df("embeddings/centroid_similarity.csv")
show_fig("centroid_similarity_heatmap.png")
show_fig("semantic_vs_accuracy_scatter.png")\
"""),
    md("_The effective-dimension and semantic-overlap evidence for benchmark redundancy._"),
]

NOTEBOOKS = {
    "00_overview.ipynb": NB00,
    "01_results_table.ipynb": NB01,
    "02_judge_agreement.ipynb": NB02,
    "03_gold_errors_verification.ipynb": NB03,
    "04_capability_coverage.ipynb": NB04,
    "05_redundancy_correlation_embeddings.ipynb": NB05,
}

KERNEL_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}


def to_source_lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines or [""]


def build_notebook(cells: list[dict]) -> dict:
    out_cells = []
    for i, c in enumerate(cells):
        cell = dict(c)
        cell["id"] = f"cell-{i:02d}"  # stable id (nbformat>=5.1 requires one)
        cell["source"] = to_source_lines(cell["source"])
        out_cells.append(cell)
    return {"cells": out_cells, "metadata": KERNEL_META, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        nb = build_notebook(cells)
        (HERE / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {name} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
