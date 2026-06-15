# Benchmark Scores Are Pipeline-Dependent: A Reliability Audit of Cybersecurity LLM Benchmarks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

LLM benchmark scores are often treated as stable measurements of model capability, yet their outcomes depend on configurable evaluation pipelines. We audit the reliability of **eight cybersecurity benchmark families** across **10 frontier, open-weight, and cybersecurity-specialized LLMs**. By modeling benchmarks as measurement pipelines, we identify **15 recurring failure modes** and show that single pipeline choices can shift scores by over 80 percentage points and substantially alter model rankings. Using a unified evaluation harness that standardizes pipeline choices while preserving task semantics, we find that **7 of 10 models shift by at least three ranks** on at least one benchmark.

---

## Measurement Pipeline Framework

We model each benchmark as a five-stage measurement pipeline:

$$\mathcal{S}_b(m) = \mathcal{A}_b(\mathcal{E}_b(\mathcal{I}_b(m, \mathcal{P}_b(\mathcal{D}_b))))$$

| Stage | Symbol | Description |
|-------|--------|-------------|
| **Dataset construction** | $\mathcal{D}$ | Task coverage, label quality, answer representation |
| **Prompt specification** | $\mathcal{P}$ | Instructions, chat formatting, output-format constraints |
| **Inference configuration** | $\mathcal{I}$ | Decoding, token budgets, stop sequences, serving backends |
| **Extraction & scoring** | $\mathcal{E}$ | Converting outputs to predictions or graded judgments |
| **Aggregation** | $\mathcal{A}$ | Combining item- and task-level scores into benchmark scores |

A reported score $\mathcal{S}_b(m)$ is not an intrinsic property of the model; it is conditional on the full pipeline used to produce it.

### Failure Modes Identified

| Failure | Description | Observed impact | Mitigation |
|---------|-------------|----------------|------------|
| $\mathcal{F}_1(\mathcal{D})$ | Limited capability coverage | 4/8 benchmarks contain ≥97% knowledge-oriented items | Stratify scores by K/A; require minimum analytical fraction |
| $\mathcal{F}_2(\mathcal{D})$ | Gold-label correctness | 23.9% confirmed label errors among manually verified flagged items | Search-grounded verifier on tier-1/2 whitelist |
| $\mathcal{F}_1(\mathcal{P})$ | Format-token leakage | ~90-point gap from literal answer-template copying | Replace format placeholders with unambiguous instructions that cannot be mistaken for the answer |
| $\mathcal{F}_2(\mathcal{P})$ | Prompt–question conflict | Up to 33% of multi-answer items answered as single-answer | Audit item wording for consistency with benchmark-level instruction; flag items whose phrasing implies fewer answers than the gold label |
| $\mathcal{F}_3(\mathcal{P})$ | Template incompatibility | 48-point gap from chat-template mismatch | Apply model-native chat template; ablate temperature and format compliance separately |
| $\mathcal{F}_1(\mathcal{I})$ | Stop-sequence mismatch | **86-point swing** from one stop-sequence parameter | Apply stop sequences post-generation after extracting from `</think>`; specify stop-sequence behaviour separately per reasoning architecture |
| $\mathcal{F}_2(\mathcal{I})$ | Token-budget filter | **81-point recovery** after increasing token budget | Set `max_tokens` ≥ 16; validate against all target API backends before release |
| $\mathcal{F}_3(\mathcal{I})$ | Temperature drift | 40-point gap between documented and enacted decoding | Reproduce paper-prescribed temperature; pin decoding config in the evaluation script |
| $\mathcal{F}_1(\mathcal{E})$ | Extractor divergence | ~88-point gap from regex/output mismatch | Designate a single canonical extractor; link it prominently to discourage reimplementations |
| $\mathcal{F}_2(\mathcal{E})$ | Denominator inflation | Up to **×500 inflation** under valid-only scoring | Report both correct/valid and correct/total; flag models with invalid rate above a documented threshold |
| $\mathcal{F}_3(\mathcal{E})$ | Metric-direction mismatch | Up to five-rank inversion on related scoring tasks | Standardise on $1{-}\text{MAD}/R$ and document $R$ explicitly |
| $\mathcal{F}_4(\mathcal{E})$ | Reasoning–extraction conflict | Up to 40-point prompt-mode sensitivity | Use LLM judge or full-body extraction |
| $\mathcal{F}_1(\mathcal{A})$ | Logprob vs. generative scoring | Up to 41-point gap on identical items | Use logprob scoring for MCQ; document scoring method; do not mix scoring modes across models |
| $\mathcal{F}_2(\mathcal{A})$ | Task-level metric drift | Up to 70-point gap from partial-credit rules | Report both Correct and Plausible Accuracy |
| $\mathcal{F}_3(\mathcal{A})$ | Aggregation inconsistency | Up to 100-point discrepancy under mixed denominators | Define accuracy = correct/total throughout |

---

## Evaluated Benchmarks

23 sub-tasks across 8 benchmark families (107,592 items total):

| Benchmark | Sub-tasks | Items | Domain | Type |
|-----------|-----------|------:|--------|------|
| MMLU-CS | computer_security | 100 | General computer security | MCQ |
| SecEval | — | 2,189 | Broad cybersecurity (9 domains) | MCQ |
| SECURE | MAET, CWET, KCV | 4,066 | ICS / OT Security | MCQ |
| CTI-Bench | MCQ, RCM, VSP, ATE, TAA | 4,947 | Cyber Threat Intelligence | MCQ + SAQ |
| AthenaBench | CKT, ATE, RCM, RMS, VSP, TAA | 8,100 | Cyber Threat Intelligence | MCQ + SAQ |
| CyberMetric | — | 10,000 | Broad cybersecurity (9 domains) | MCQ |
| RedSage-Bench | Frameworks, Generals, Skills, CLI, Kali | 30,280 | Tool & framework proficiency | MCQ |
| SecBench | English MCQ | 47,910 | Multi-dimensional cybersecurity | MCQ |

See [unified-benchmark-pipeline/README.md](unified-benchmark-pipeline/README.md) for the exact `--tasks` flag names.

### Adoption by Cybersecurity-Specialized LLMs

| Benchmark | Primus ([2502.11191](https://arxiv.org/abs/2502.11191)) | RedSage ([2601.22159](https://arxiv.org/abs/2601.22159)) | Foundation-Sec-8B ([2504.21039](https://arxiv.org/abs/2504.21039)) | Sec-Gemini v1 |
|-----------|:---:|:---:|:---:|:---:|
| CTI-Bench | ✓ (MCQ, RCM, VSP, ATE) | ✓† (MCQ, RCM) | ✓† (MCQ, RCM) | ✓† (MCQ, RCM) |
| AthenaBench | — | — | — | — |
| SECURE | — | ✓ (MAET, CWET, KCV) | ✗ saturated | — |
| SecEval | ✓ | ✓ | ✗ saturated | — |
| CyberMetric | ✓ | ✓ | ✓ | — |
| SecBench | — | ✓ | ✓ (English MCQ) | — |
| MMLU-CS | — | ✓ | ✗ n=100 too small | — |
| RedSage-Bench | — | ✓ (custom) | — | — |

---

## Evaluated Models

10 LLMs across three categories:

| Model | Category |
|-------|---------|
| GPT-5.4 | Proprietary |
| Claude Sonnet 4.6 | Proprietary |
| Gemma-4-31B | Open-weight |
| Qwen3.6-35B | Open-weight |
| Llama-3.3-70B | Open-weight |
| GPT-OSS-20B | Open-weight |
| Primus-Nemotron-70B | Cybersecurity-specialized |
| Primus-Merged-8B | Cybersecurity-specialized |
| Foundation-Sec-8B | Cybersecurity-specialized |
| RedSage-Qwen3-8B-DPO | Cybersecurity-specialized |

---

## Repository Structure

```
BenchmarkingSecBenchmarks/
├── unified-benchmark-pipeline/     # Standardized harness (inference → judge → analysis)
│   ├── run_inference_benchmarks.py
│   ├── run_evaluate_llm_judge.py
│   ├── analyze_gold_errors.py
│   ├── calibrate_max_tokens.py
│   ├── experiment/
│   ├── analysis/                   # Re-runnable post-judge analysis modules
│   └── notebooks/                  # Walkthrough notebooks NB00–NB05
├── original-pipeline-exp/          # Benchmark-faithful original evaluation pipelines
│   ├── evaluate.py                 # Combined inference + regex evaluation
│   ├── run_inference_benchmarks.py # Benchmark-faithful response collection
│   ├── run_evaluate_llm_judge.py
│   ├── run_batch_evaluate.py
│   ├── run_prompt_sensitivity.py   # Prompt sensitivity study (zero-shot / few-shot / CoT)
│   ├── run_redsage_lighteval.py    # RedSage logprob scoring via upstream LightEval
│   └── notebooks/                  # Walkthrough notebooks NB00–NB05
├── external/
│   └── RedSage/                    # Git submodule: RedSage benchmark + LightEval eval harness
├── sayf-eval/                      # Git submodule: shared evaluation utilities
├── Dockerfile
└── requirements.txt
```

---

## Two Pipelines

The audit depends on running **both** pipelines and comparing their outputs.

### Original pipeline — [`original-pipeline-exp/`](original-pipeline-exp/)

Reproduces each benchmark's published evaluation logic as closely as possible: the same prompts, the same regex extractor, the same decoding parameters. This is what exposes the 15 failure modes — by running models under their own published protocols, we observe the artifacts each protocol introduces.

See [original-pipeline-exp/README.md](original-pipeline-exp/README.md) for per-script details, prompt templates, and failure-mode cross-references.

**Walkthrough notebooks** → [`original-pipeline-exp/notebooks/`](original-pipeline-exp/notebooks/)

| Notebook | Paper section |
|----------|--------------|
| [00_overview.ipynb](original-pipeline-exp/notebooks/00_overview.ipynb) | §3 — Model inventory, output tree, run order |
| [01_results_table.ipynb](original-pipeline-exp/notebooks/01_results_table.ipynb) | Table 2 — Full score table; corrected GPT-5.4 SecEval score (81.4%) |
| [02_failure_modes.ipynb](original-pipeline-exp/notebooks/02_failure_modes.ipynb) | Table 3 — 15 failure modes headline numbers |
| [03_prompt_sensitivity.ipynb](original-pipeline-exp/notebooks/03_prompt_sensitivity.ipynb) | §4.2 — Zero-shot / few-shot / CoT sensitivity (F1–F3(P)) |
| [04_inference_config.ipynb](original-pipeline-exp/notebooks/04_inference_config.ipynb) | §4.3 — Stop-sequence, token-budget, temperature (F1–F3(I)) |
| [05_logprob_vs_generative.ipynb](original-pipeline-exp/notebooks/05_logprob_vs_generative.ipynb) | §4.5 — Logprob vs. generative; TAA metric drift (F1–F2(A)) |

### Unified pipeline — [`unified-benchmark-pipeline/`](unified-benchmark-pipeline/)

A standardized harness that applies consistent prompt formatting, a single LLM-as-judge extractor, and fixed decoding parameters across all benchmarks and models. Running the same model outputs through both pipelines is how we measure the rank shifts reported in Table 4.

See [unified-benchmark-pipeline/README.md](unified-benchmark-pipeline/README.md) for the full workflow.

**Walkthrough notebooks** → [`unified-benchmark-pipeline/notebooks/`](unified-benchmark-pipeline/notebooks/)

| Notebook | Paper section |
|----------|--------------|
| [00_overview.ipynb](unified-benchmark-pipeline/notebooks/00_overview.ipynb) | §3 — Pipeline overview, task inventory |
| [01_results_table.ipynb](unified-benchmark-pipeline/notebooks/01_results_table.ipynb) | Table 4 — Unified score table and rank comparison |
| [02_judge_agreement.ipynb](unified-benchmark-pipeline/notebooks/02_judge_agreement.ipynb) | §4.4 — LLM judge vs. regex agreement rates |
| [03_gold_errors.ipynb](unified-benchmark-pipeline/notebooks/03_gold_errors.ipynb) | §4.1 — Gold-label error analysis (F2(D)) |
| [04_capability_coverage.ipynb](unified-benchmark-pipeline/notebooks/04_capability_coverage.ipynb) | §4.1 — K/A stratification; PCA (95.3% PC1 variance) |
| [05_redundancy_correlation_embeddings.ipynb](unified-benchmark-pipeline/notebooks/05_redundancy_correlation_embeddings.ipynb) | §5 — Cross-benchmark rank correlation (τ=0.29/0.24); 7/10 rank shifts ≥3 |

### Pipeline comparison

| Concern | Original pipeline | Unified pipeline |
|---------|-----------------|-----------------|
| Inference | `original-pipeline-exp/run_inference_benchmarks.py` | `unified-benchmark-pipeline/run_inference_benchmarks.py` |
| Scoring | Regex extraction (benchmark-faithful) | LLM-as-judge |
| Decoding params | Benchmark-prescribed (may conflict) | Standardized |
| Prompt templates | Benchmark-prescribed | Standardized |
| Purpose | Measure failure modes | Measure rank shifts under controlled evaluation |

---

## Submodules

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/qcri/BenchmarkingSecBenchmarks.git

# Or initialize after a plain clone
git submodule update --init --recursive
```

### `external/RedSage`

The RedSage benchmark repository. Contains the LightEval-based evaluation harness used for logprob scoring in `original-pipeline-exp/run_redsage_lighteval.py`. The logprob vs. generative comparison (F1(A)) depends on the upstream implementation in `external/RedSage/eval/cybersecurity_benchmarks.py`.

```bash
# Required one-time setup for RedSage logprob scoring
cd external/RedSage/eval/lighteval && pip install -e . && cd ../../../../
pip install cvss aenum
```

### `sayf-eval`

Shared evaluation utilities used by the unified pipeline's analysis modules.

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- CUDA-compatible GPU with ≥16 GB VRAM (for local open-weight inference)
- Azure OpenAI or Anthropic API key (for proprietary models and the LLM judge)

```bash
git clone --recurse-submodules https://github.com/qcri/BenchmarkingSecBenchmarks.git
cd BenchmarkingSecBenchmarks

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Containerized Setup

```bash
docker build -t secbench-gpu .
docker run --gpus all -it secbench-gpu
```

### Run the Unified Pipeline

```bash
cd unified-benchmark-pipeline
cp .env.example .env  # fill in AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME

# 1. Collect responses
python run_inference_benchmarks.py \
  --model_path "meta-llama/Llama-3.3-70B-Instruct" \
  --use_vllm \
  --tasks mcq rcm vsp ate seceval cybermetric secbench \
  --output_dir outputs/responses_llama33_70b

# 2. Judge responses
python run_evaluate_llm_judge.py \
  --response_dir outputs/responses_llama33_70b \
  --judge_use_api \
  --judge_api_endpoint "$AZURE_ENDPOINT" \
  --judge_api_model "$AZURE_DEPLOYMENT_NAME" \
  --output outputs/judge_llama33_70b/eval_results

# 3. Run post-judge analysis
cd analysis
PYTHONPATH=. python -m analysis.results_table
PYTHONPATH=. python -m analysis.judge_agreement
```

### Run the Original Pipeline (benchmark-faithful)

```bash
cd original-pipeline-exp

# Combined inference + regex evaluation (replicates benchmark-published behavior)
python evaluate.py \
  --model_path "meta-llama/Llama-3.3-70B-Instruct" \
  --tasks mcq rcm vsp ate seceval cybermetric \
  --output eval_results.json

# Prompt sensitivity study (F1(P)–F3(P))
python run_prompt_sensitivity.py \
  --model_path "meta-llama/Llama-3.3-70B-Instruct" \
  --use_vllm \
  --tasks mcq rcm vsp ate \
  --prompt_modes zero_shot few_shot cot \
  --n_test 100 --output_dir outputs/prompt_sensitivity

# RedSage logprob scoring vs. generative (F1(A))
python run_redsage_lighteval.py \
  --model RISys-Lab/RedSage-Qwen3-8B-DPO
```

### Browse the Notebooks

```bash
# Original pipeline notebooks
cd original-pipeline-exp/notebooks
pip install -r requirements-notebooks.txt
jupyter lab

# Unified pipeline notebooks
cd unified-benchmark-pipeline/notebooks
jupyter lab
```

To regenerate heavy cells (GPU inference, API calls):

```bash
SAYF_NB_REGEN=1 jupyter nbconvert --to notebook --execute 01_results_table.ipynb
```

---

## Hardware

All open-weight and cybersecurity-specialized models are evaluated on a single server node with one **NVIDIA H200 SXM 141 GB GPU**, served via vLLM. Proprietary models (GPT-5.4, Claude Sonnet 4.6) are accessed via their respective APIs.

---

## Troubleshooting

**Out of memory:** Use fewer tasks, reduce `--max_samples`, or switch to an API judge.

**vLLM instability:** Omit `--use_vllm` / `--judge_use_vllm` to fall back to HuggingFace Transformers.

**Submodules missing:** Run `git submodule update --init --recursive`.

**Long-running jobs:**

```bash
screen -S eval
python run_inference_benchmarks.py ...
# Detach: Ctrl-A D  |  Reattach: screen -r eval
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
