# Paper Experiments & Visualizations

This directory contains 7 Python scripts that automate the experiments and visualizations needed to complete the paper sections marked as "pending" or "placeholder-driven" in the draft.

## Pipeline Architecture

```
run_all_experiments.py (Master Orchestrator)
    │
    ├─→ 01_full_scale_evaluation.py (Main Evaluation)
    │       ├─→ run_inference_benchmarks.py (7 models × 21 tasks)
    │       └─→ run_evaluate_llm_judge.py (LLM judge evaluation)
    │
    ├─→ 02_risys_cluster_analysis.py (uses results from #1)
    ├─→ 03_ate_protocol_analysis.py (uses results from #1)
    ├─→ 04_secure_ceiling_analysis.py (uses results from #1)
    ├─→ 05_backend_variance_analysis.py (uses results from #1)
    ├─→ 06_generate_framework_visualizations.py (creates plots)
    └─→ 07_generate_reports.py (creates markdown reports)
```

**Key Principle:** Script #1 does the heavy lifting (inference + evaluation). Scripts 2-7 analyze the results.

## Overview

The scripts address the incomplete sections in the paper by generating:
1. **Full-scale empirical evaluation** (Section 5)
2. **RISys-Lab cluster circularity analysis** (Section 4.1)
3. **ATE format sensitivity analysis** (Section 4.2)
4. **SECURE ceiling effect evidence** (Section 4.3)
5. **Backend variance documentation** (Section 4.4)
6. **Framework evaluation visualizations** (Table 2, figures)
7. **Comprehensive practitioner guides** (Section 7 + appendix)

## Quick Start

### Master Orchestrator (Recommended)

**run_all_experiments.py** - Runs all 7 scripts in sequence:

```bash
# Full-scale evaluation (takes 5-9 hours for 7 models × 21 tasks)
python run_all_experiments.py

# Test on small subset first (recommended - 10 samples per task)
python run_all_experiments.py --subset

# Skip inference collection, reuse cached results (runs scripts 2-7 only)
python run_all_experiments.py --skip-inference

# Run only specific analysis
python run_all_experiments.py --only 2  # Run RISys analysis (script #2)
```

### What --subset does:
- Runs 10 samples per task instead of full datasets
- Validates entire pipeline (all 7 models × 21 tasks)
- Completes in ~1-2 hours
- **Use this first** to test before full-scale run

### Run Individual Scripts

```bash
# Full-scale evaluation (takes 2-6 hours depending on hardware)
python run_all_experiments.py

# Test on small subset first (5-10 minutes)
python run_all_experiments.py --subset

# Skip inference collection, reuse cached results
python run_all_experiments.py --skip-inference

# Run only specific analysis
python run_all_experiments.py --only 2  # Run RISys analysis (script #2)
```

### Run Individual Scripts

```bash
# Full-scale evaluation (all models, all 21 tasks, 100+ samples each)
python 01_full_scale_evaluation.py

# Analyze if RedSage scores higher on its own benchmarks
python 02_risys_cluster_analysis.py

# Test ATE evaluation protocol across multiple models
python 03_ate_protocol_analysis.py

# Baseline testing to detect SECURE ceiling effects
python 04_secure_ceiling_analysis.py

# Compare HuggingFace vs vLLM backend variance
python 05_backend_variance_analysis.py

# Generate framework heatmap and visualizations
python 06_generate_framework_visualizations.py

# Generate practitioner guides and recommendations
python 07_generate_reports.py
```

## Scripts in Detail

### 1. `01_full_scale_evaluation.py`

**Purpose:** Complete empirical evaluation across all models and tasks (Figure)

**What it does:**
- Collects inferences from 7 models on all 21 tasks
- Models: Llama-3.1-8B (base), Llama-Primus-Merged, Llama-Primus-Base, Foundation-Sec-8B, RedSage-8B-Ins, RedSage-8B-DPO, Fanar-AR
- Tasks: Full datasets per task (CTI-Bench, AthenaBench, SECURE, SecBench, etc.)
- Uses LLM judge (Llama-3.1-8B) with HuggingFace backend
- Computes confidence intervals (95% CI)
- Runs statistical tests between model pairs

**Pipeline:**
1. Calls `run_inference_benchmarks.py` for each model (Step 1/4)
2. Calls `run_evaluate_llm_judge.py` for each model (Step 2/4)
3. Computes statistical tests (Step 3/4)
4. Generates summary tables (Step 4/4)

**Output:**
- `full_scale_results.json` - All results with confidence intervals
- `model_comparison_table.csv` - Easy import to Excel/paper

**For paper:** Completes Section 5 (Empirical Evaluation)

**Runtime:**
- Full-scale: ~5-9 hours (7 models × 21 tasks × full datasets)
- Subset mode (`--subset`): ~1-2 hours (7 models × 21 tasks × 10 samples)
- Can run in parallel on multiple GPUs

---

### 2. `02_risys_cluster_analysis.py`

**Purpose:** Quantify circular evaluation evidence (Addresses Section 4.1)

**What it does:**
- Extracts performance on RISys-Lab benchmarks (CTI-Bench, AthenaBench, SECURE)
- Extracts performance on independent benchmarks (CyberMetric, SecEval, CISSP)
- Compares RedSage vs other models:
  - Does RedSage have disproportionate advantage on own benchmarks?
  - Quantifies: `advantage_on_risys - advantage_on_independent`
- Tests hypothesis: >15% higher performance on own benchmarks = circular evaluation

**Output:**
- `risys_cluster_analysis.json` - Structured analysis
- `risys_circular_evidence.md` - Human-readable report

**For paper:** Provides quantitative evidence for Section 4.1 claim

**Critical finding:**
- If RedSage outperforms PRIMUS by X% on RISys benchmarks
- But only outperforms by X-10% on independent benchmarks
- → Suggests benchmark-specific tuning

---

### 3. `03_ate_protocol_analysis.py`

**Purpose:** Demonstrate ATE format sensitivity bug (Section 4.2)

**What it does:**
- Simulates diverse model response formats:
  - Numbered lists: "1. T1059\n2. T1055"
  - Prose: "The techniques are T1059 and T1055"
  - Bullet points: "• T1059"
- Evaluates with strict regex (original): ~0% F1
- Evaluates with normalized extraction: matches correctly
- Shows 30-40% of responses fail due to format mismatch

**Output:**
- `ate_protocol_analysis.json` - Format sensitivity metrics
- `ate_response_examples.txt` - Raw model output examples
- `ate_format_sensitivity.md` - Analysis report

**For paper:** Proves ATE bug affects all models systematically

**Key stat:** "X% of responses are semantically correct but format-rejected"

---

### 4. `04_secure_ceiling_analysis.py`

**Purpose:** Quantify SECURE ceiling effects with multiple baselines (Section 4.3)

**What it does:**
- Tests SECURE tasks (MAET, CWET, KCV) on model baselines of varying capability:
  - Weak: Mistral-7B, Llama-2-7B
  - Standard: Llama-3.1-8B-Instruct
  - Strong: Llama-3.1-70B-Instruct
  - Specialized: PRIMUS, RedSage
- Checks if performance saturates at 1.0 or varies by capability
- If all models hit 100%, tasks have no discriminative power

**Output:**
- `secure_ceiling_analysis.json` - Results by model tier
- `secure_model_performance.csv` - Performance matrix
- `secure_ceiling_evidence.md` - Interpretation

**For paper:** Confirms ceiling effects (D1 concern)

**Expected findings:**
- MAET, CWET, KCV: all models → 1.0 (ceiling confirmed)
- **Recommendation:** Exclude from aggregate scores

---

### 5. `05_backend_variance_analysis.py`

**Purpose:** Document backend-induced variance (Section 4.5)

**What it does:**
- Runs same evaluation with HF Transformers backend
- Runs same evaluation with vLLM backend
- Tests HF reproducibility: two runs produce identical results?
- Measures vLLM accuracy drop (targets: 57.89% → 47.50%)
- Diagnoses vLLM NCCL deadlock failures

**Output:**
- `backend_variance_analysis.json` - Variance metrics
- `backend_comparison.csv` - Side-by-side comparison
- `backend_variance_report.md` - Root cause analysis

**For paper:** Documents reproducibility failures

**Key stat:** "vLLM reduces accuracy 10.39 percentage points despite identical responses"

---

### 6. `06_generate_framework_visualizations.py`

**Purpose:** Create visual representations of Table 2 (Framework scoring matrix)

**What it does:**
- Generates heatmap: benchmarks (rows) × dimensions (cols)
  - H=High (green), M=Medium (yellow), L=Low (orange), N=None (red)
- Scatter plot: benchmark size vs validation provenance
- Bar chart: dimension coverage distribution
- Interactive HTML version (requires plotly)

**Output:**
- `framework_heatmap.png` - Main heatmap visualization
- `framework_scatter.png` - Size vs validation
- `framework_dimensions.png` - Coverage by dimension
- `framework_interactive.html` - Interactive dashboard
- `framework_summary.json` - Structured data

**For paper:** Figures 1-2 and Table 2 visualizations

---

### 7. `07_generate_reports.py`

**Purpose:** Generate practitioner guides and recommendation matrices

**What it does:**
- Creates **Practitioner's Guide**: "Which benchmark should I use?"
- Lists use cases → recommended benchmarks mapping
- Provides critical issues table (what to watch out for)
- Generates evaluation checklist (for model paper authors)
- Creates author improvement roadmap (for benchmark creators)

**Output:**
- `practitioner_guide.md` - 2-page practical guide
- `benchmark_recommendations.csv` - Benchmark quality rankings
- `model_eval_checklist.md` - Pre-publication checklist
- `author_improvement_roadmap.md` - How to fix benchmarks

**For paper:** Sections 7 (Protocol) + Appendix (Recommendations)

## Output Structure

All scripts generate results in `./results/`:

```
results/
├── full_scale_results.json
├── model_comparison_table.csv
├── risys_cluster_analysis.json
├── risys_circular_evidence.md
├── ate_protocol_analysis.json
├── ate_response_examples.txt
├── ate_format_sensitivity.md
├── secure_ceiling_analysis.json
├── secure_model_performance.csv
├── secure_ceiling_evidence.md
├── backend_variance_analysis.json
├── backend_comparison.csv
├── backend_variance_report.md
├── framework_heatmap.png
├── framework_scatter.png
├── framework_dimensions.png
├── framework_interactive.html
├── framework_summary.json
├── practitioner_guide.md
├── benchmark_recommendations.csv
├── model_eval_checklist.md
└── author_improvement_roadmap.md
```

## Requirements

### Python Packages

```bash
pip install numpy scipy pandas matplotlib seaborn plotly transformers datasets peft tqdm
```

### Optional for faster inference:
```bash
pip install vllm  # For backend variance testing
```

### GPU Requirements

- Minimum: 1x GPU with 24GB+ VRAM (for Llama-3.1-8B judge)
- Recommended: 2+ GPUs for parallel inference

### Disk Space

- ~100GB for model weights (cached by HuggingFace)
- ~5GB for benchmark datasets
- ~2GB for results + visualizations

## Configuration

Edit variables in each script to customize:

```python
# In 01_full_scale_evaluation.py
MODELS = [...]  # Add/remove models
TASKS = [...]   # Change task selection

# In 02_risys_cluster_analysis.py
critical_threshold = 0.15  # Circular eval threshold (15%)

# In 06_generate_framework_visualizations.py
SCORES = {...}  # Update framework scores if needed
```

## Running on Remote Servers

### SLURM Cluster Example

```bash
#!/bin/bash
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --mem=200G

cd /path/to/experiment
python run_all_experiments.py
```

### Using Screen or Tmux

```bash
screen -S experiments
python run_all_experiments.py

# Detach: Ctrl-A, then D
# Reattach: screen -r experiments
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size in run_evaluate_llm_judge.py or use smaller judge model
python 01_full_scale_evaluation.py --batch-size 4 --judge-model "meta-llama/Llama-2-7B-Instruct"
```

### vLLM Errors

```bash
# vLLM often fails; switch to HuggingFace backend in specific scripts
# Edit: change judge_backend = "huggingface" in scripts
```

### Missing Model Weights

```bash
# Automatically downloaded on first use; ensure ~500GB free space
# Or pre-download:
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')"
```

## Expected Runtime

| Script | Models | Tasks | Time | Notes |
|--------|--------|-------|------|-------|
| 01_full_scale | 7 | 21 | 5-9 hrs | Main evaluation (parallelizable) |
| 02_risys | — | — | <1 min | Uses cached results from #1 |
| 03_ate | — | — | <1 min | Simulated data |
| 04_secure | 8 | 3 | 1-2 hrs | Multiple baselines |
| 05_backend | — | 5 | 30 min | HF + vLLM comparison |
| 06_visualizations | — | — | <1 min | Matplotlib/Seaborn |
| 07_reports | — | — | <1 min | Text generation |

**With --subset flag:** Script #1 takes ~1-2 hours (10 samples per task)
| 07_reports | — | — | <1 min | Text generation |

## Integration with Paper

### For each section:

**Section 4 (Gap Analysis):**
- Use `ate_protocol_analysis.md` (4.2)
- Use `secure_ceiling_evidence.md` (4.3)
- Use `backend_variance_report.md` (4.4)
- Use `risys_circular_evidence.md` (4.1)

**Section 5 (Empirical Evaluation):**
- Use `full_scale_results.json` for Table 5 (main empirical table)
- Use `model_comparison_table.csv` in LaTeX

**Figures:**
- Figure X: `framework_heatmap.png` (Table 2 visualization)
- Figure Y: `framework_scatter.png` (benchmark scale vs quality)
- Figure Z: `framework_dimensions.png` (coverage distribution)

**Section 7 (Protocol):**
- Use `model_eval_checklist.md` (requirements checklist)

**Appendix:**
- Include `practitioner_guide.md`
- Include `author_improvement_roadmap.md`
- Include `benchmark_recommendations.csv`

## Contributing & Extending

To add new analyses:

1. Create `08_your_analysis.py` following the template pattern
2. Add to `SCRIPTS` list in `run_all_experiments.py`
3. Update this README with script description
4. In your script: use `OUTPUT_DIR = Path("results")` for consistency

## License

Same as main paper repository.

## Questions?

Refer to the inline documentation in each script, or check the generated `.md` reports for interpretation guidance.
