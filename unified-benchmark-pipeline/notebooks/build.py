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
| `02_judge_agreement` | LLM-judge reliability + independence (Cohen's κ; second-judge cross-check) | 𝓔 extraction/scoring |
| `03_gold_errors_verification` | Suspect gold labels + verification + two-annotator spot-check | 𝐹₂(𝒟) label quality |
| `04_capability_coverage` | Knowledge-vs-Analytical coverage | 𝐹₁(𝒟) limited coverage |
| `05_redundancy_correlation_embeddings` | Cross-task redundancy, effective dimensions | benchmark redundancy |
| `06_rank_shift_stability` | Rank-shift bootstrap CIs + generation-vs-extraction decomposition | ranking stability |
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
    md("""\
### Judge independence — a second, different-family judge

The pinned judge (GPT-5.4) is also one of the ten evaluated models, so its
verdicts could in principle favor its own family. We re-graded every stored
output with an independent judge (Claude Sonnet 4.6) and compared verdicts, and
additionally stressed the judge on the two strata where it can fail: calls it
returned `NONE`, and MCQ-family calls where its verdict differs from a
deterministic regex (`analysis.judge_adversarial`).
"""),
    code("""\
import json, pandas as pd
from IPython.display import display
s = json.loads((REPORTS_DIR / "judge_adversarial/summary.json").read_text())
m = s["machine_second_rater_gpt_vs_claude"]
display(pd.DataFrame([
    {"stratum": k, "n": v["n"], "agreement_%": v["agreement_pct"], "cohens_kappa": v["cohens_kappa"]}
    for k, v in m.items()
]))
print("adversarial strata sizes:", s["strata_sizes"])\
"""),
    md("""\
The two independent judges agree **99.6%** on the full set (κ≈0.99) and ~98% even
on the adversarial strata; GPT-5.4 gains only +0.22 pp from its own-family judge,
and the leaderboard is unchanged (Spearman ρ≈0.98). Two human annotators then
adjudicated a 120-item adversarial sample.
"""),
    code("""\
import pandas as pd
adv = pd.read_csv(REPORTS_DIR / "judge_adversarial/adversarial_sheet.csv")
print("adversarial items:", len(adv), " strata:", adv["stratum"].value_counts().to_dict())
if {"annotator_A", "annotator_B"}.issubset(adv.columns):
    both = adv[(adv["annotator_A"].astype(str).str.strip() != "") &
               (adv["annotator_B"].astype(str).str.strip() != "")]
    if len(both):
        print(f"human-annotated: {len(both)}/{len(adv)}  raw agreement: "
              f"{(both['annotator_A'] == both['annotator_B']).mean():.3f}")
    adj = adv.get("adjudicated")
    if adj is not None:
        adj = adj.astype(str).str.strip(); adj = adj[adj != ""]
        if len(adj):
            print(f"adjudicated judge accuracy: {(adj == 'judge_correct').sum()}/{len(adj)}")\
"""),
    md("_Judge-independence and adversarial-reliability evidence (paper App.~B.5)._"),
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
    md("""\
### Human two-annotator label spot-check

To calibrate the automated verifier we drew a 50-item sample and had two
annotators independently judge whether they *agree* with each verifier verdict
(Cohen's κ + the adjudicated confirmation rate). This is the human validation the
label audit reports.
"""),
    code("""\
import pandas as pd
sp = pd.read_csv(REPORTS_DIR / "label_spotcheck/spotcheck.csv")
print("items:", len(sp), " verifier verdicts:", sp["verifier_verdict"].value_counts().to_dict())
if {"annotator_A", "annotator_B"}.issubset(sp.columns):
    both = sp[(sp["annotator_A"].astype(str).str.strip() != "") &
              (sp["annotator_B"].astype(str).str.strip() != "")]
    if len(both):
        print(f"human-annotated: {len(both)}/{len(sp)}  raw agreement: "
              f"{(both['annotator_A'] == both['annotator_B']).mean():.3f}")
    adj = sp.get("adjudicated")
    if adj is not None:
        adj = adj.astype(str).str.strip(); adj = adj[adj != ""]
        if len(adj):
            print("adjudicated:", adj.value_counts().to_dict())\
"""),
    md("""\
Two annotators confirm the verifier on **46 of 50** verdicts (Cohen's κ≈0.68); the
four exceptions are two identified verifier errors and two items unverifiable from
the cited sources — the grounded-search verifier is right on the large majority of
the sample without being infallible.
"""),
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

# ─────────────── 06 — rank-shift stability & decomposition ────────────────
NB06 = [
    md("""\
# 06 · Rank-shift stability & decomposition

Standardizing the pipeline reorders the leaderboard (paper Table 5). Two
questions follow: **(1)** are the shifts real or sampling noise? — an item-level
**bootstrap**; and **(2)** is the shift driven by the LLM judge or by the other
standardizations? — a **generation-vs-extraction decomposition**. Both re-score
stored outputs; no new generation.
"""),
    setup_cell(),
    md("""\
### Item-level bootstrap CIs — `analysis.bootstrap_ci`  *(heavy · re-scores stored per-item outputs)*
Rendered from cached `bootstrap_ci.json`. `SAYF_NB_REGEN=1` recomputes (5,000
paired resamples of the stored per-item scores). ρ_b is Spearman's ρ between the
original and standardized per-benchmark rankings.
"""),
    code("""\
if heavy("bootstrap_ci"):
    import importlib; importlib.import_module("analysis.bootstrap_ci")
import json, pandas as pd
from IPython.display import display
b = json.loads((REPORTS_DIR / "bootstrap_ci.json").read_text())
rows = []
for name, e in b["benchmarks"].items():
    lo, hi = e["rho_ci"]
    rows.append({"benchmark": name, "rho_b": round(e["rho_point"], 2),
                 "ci_low": round(lo, 2), "ci_high": round(hi, 2)})
display(pd.DataFrame(rows))
print(f"{b['B']} resamples; every interval excludes 1, only SecEval includes 0")\
"""),
    md("""\
### Generation vs. extraction — `analysis.decomp_full`  *(heavy · reads original + standardized generations)*
Rendered from cached `decomp_full.json`. **GEN** = ρ(regex@original → regex@standardized)
(generation change, extractor held at regex); **EXT** = ρ(regex@standardized →
judge@standardized) (extractor swap, generation fixed). Lower ρ = more reordering;
the ρ(A',A) column checks the regex reproduces the published *before* ranking.
"""),
    code("""\
if heavy("decomp_full"):
    import importlib; importlib.import_module("analysis.decomp_full")
import json, pandas as pd
from IPython.display import display
d = json.loads((REPORTS_DIR / "decomp_full.json").read_text())
rows = []
for name, e in d["benchmarks"].items():
    rows.append({"benchmark": name,
                 "GEN rho(A'->B')": e["GEN_rho_Ap_Bp"],
                 "EXT rho(B'->C)": e["EXT_rho_Bp_C"],
                 "check rho(A',A)": e["rho_Ap_A_regex_reproduces_before"]})
display(pd.DataFrame(rows))\
"""),
    md("""\
The extraction (judge) swap co-drives the shift only on **MMLU-CS** (EXT ρ≈0.20)
and mildly SECURE; on AthenaBench, CTI-Bench and SecBench the judge barely reorders
(EXT ρ 0.99 / 0.89 / 0.61) and the **generation** change dominates. So Table 5 is
not an artifact of "replacing regexes with a judge" (paper App.~L.4/L.5; the two
attacker-attribution tasks are excluded — a regex cannot do alias matching).
"""),
]

NOTEBOOKS = {
    "00_overview.ipynb": NB00,
    "01_results_table.ipynb": NB01,
    "02_judge_agreement.ipynb": NB02,
    "03_gold_errors_verification.ipynb": NB03,
    "04_capability_coverage.ipynb": NB04,
    "05_redundancy_correlation_embeddings.ipynb": NB05,
    "06_rank_shift_stability.ipynb": NB06,
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
