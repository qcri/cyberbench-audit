# Benchmark Scores Are Pipeline-Dependent: A Reliability Audit of Cybersecurity LLM Benchmarks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


## Overview

LLM benchmark scores are often treated as stable measurements of model capability, yet their outcomes depend on configurable evaluation pipelines. We audit the reliability of **eight cybersecurity benchmark families** across **10 frontier, open-weight, and cybersecurity-specialized LLMs**. By modeling benchmarks as measurement pipelines, we identify **15 recurring failure modes** and show that single pipeline choices can shift scores by over 80 percentage points and substantially alter model rankings. Using a unified evaluation harness that standardizes pipeline choices while preserving task semantics, we find that **7 of 10 models shift by at least three ranks** on at least one benchmark.

## Measurement Pipeline Framework

We model each benchmark as a measurement pipeline with five stages:

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
| $\mathcal{F}_1(\mathcal{I})$ | Stop-sequence mismatch | 86-point swing from one stop-sequence parameter | Apply stop sequences post-generation after extracting from `</think>`; specify stop-sequence behaviour separately per reasoning architecture |
| $\mathcal{F}_2(\mathcal{I})$ | Token-budget filter | 81-point recovery after increasing token budget | Set `max_tokens` ≥ 16; validate against all target API backends before release |
| $\mathcal{F}_3(\mathcal{I})$ | Temperature drift | 40-point gap between documented and enacted decoding | Reproduce paper-prescribed temperature; pin decoding config in the evaluation script |
| $\mathcal{F}_1(\mathcal{E})$ | Extractor divergence | ~88-point gap from regex/output mismatch | Designate a single canonical extractor; link it prominently to discourage reimplementations; require evaluator details in any paper reporting scores |
| $\mathcal{F}_2(\mathcal{E})$ | Denominator inflation | Up to ×500 inflation under valid-only scoring | Report both correct/valid and correct/total; flag models with invalid rate above a documented threshold |
| $\mathcal{F}_3(\mathcal{E})$ | Metric-direction mismatch | Up to five-rank inversion on related scoring tasks | Standardise on $1{-}\text{MAD}/R$ and document $R$ explicitly; or always report raw MAD with stated direction |
| $\mathcal{F}_4(\mathcal{E})$ | Reasoning–extraction conflict | Up to 40-point prompt-mode sensitivity | Use LLM judge or full-body extraction; release reference extractor alongside metric definition |
| $\mathcal{F}_1(\mathcal{A})$ | Logprob vs. generative scoring | Up to 41-point gap on identical items | Use logprob scoring for MCQ; document scoring method; do not mix scoring modes across models |
| $\mathcal{F}_2(\mathcal{A})$ | Task-level metric drift | Up to 70-point gap from partial-credit rules | Report both Correct and Plausible Accuracy; or pin single definition with documented alias-graph handling |
| $\mathcal{F}_3(\mathcal{A})$ | Aggregation inconsistency | Up to 100-point discrepancy under mixed denominators | Define accuracy = correct/total throughout; report invalid rate and judge-agreement separately |

## Evaluated Benchmarks

### Selection Rationale

The eight benchmark families were selected to satisfy three criteria jointly:

1. **Field adoption.** Each benchmark is used by at least one published cybersecurity-specialized LLM as a primary evaluation surface. Including benchmarks that no model uses would not reflect the measurement practices the paper audits.
2. **Task diversity.** Together the benchmarks span two fundamental paradigms — multiple-choice knowledge recall (MCQ) and structured open-ended extraction (SAQ) — across eight distinct cybersecurity sub-domains.
3. **Pipeline variety.** The benchmarks differ maximally in their evaluation implementations: some provide no inference script; others bundle inference and scoring together; evaluation methods range from regex extraction to CVSS numerical comparison to alias-aware set matching. This variety is what makes a cross-benchmark audit informative.

### Coverage

23 sub-tasks across 8 benchmark families:

| Benchmark | Sub-tasks | Prompts | Domain | Type |
|-----------|-----------|--------:|--------|------|
| CTI-Bench | MCQ, RCM, VSP, ATE, TAA | 4,947 | Cyber Threat Intelligence | MCQ + SAQ |
| AthenaBench | CKT, ATE, RCM, RMS, VSP, TAA | 8,100 | Cyber Threat Intelligence | MCQ + SAQ |
| SECURE | MAET, CWET, KCV | 4,066 | ICS / OT Security | MCQ |
| SecEval | — | 2,189 | Broad cybersecurity (9 domains) | MCQ |
| CyberMetric | — | 10,000 | Broad cybersecurity (9 domains) | MCQ |
| SecBench | English MCQ | 47,910 | Multi-dimensional cybersecurity | MCQ |
| MMLU-CS | computer_security | 100 | General computer security | MCQ |
| RedSage-Bench | Frameworks, Generals, Skills, CLI, Kali | 30,280 | Tool & framework proficiency | MCQ |

See `unified-benchmark-pipeline/README.md` for the exact `--tasks` flag names.

### Adoption by Cybersecurity-Specialized LLMs

The table below maps each benchmark to the specialized LLMs that use it as a primary evaluation surface, verified against the published papers. ✓ = included; ✓† = partial sub-task coverage; ✗ = explicitly excluded; — = not mentioned.

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

**Notes:**
- AthenaBench postdates all four model papers; no specialized LLM has evaluated on it yet. It is included because it extends CTI-Bench with a different scoring convention for the same tasks — a key cross-benchmark rank-disagreement case.
- Foundation-Sec-8B explicitly excludes SECURE and SecEval (both saturated) and MMLU-CS (n=100 deemed statistically unreliable).
- Sec-Gemini v1 is an experimental model blog post, not a research paper; only CTI-MCQ and CTI-RCM results are reported.

### Why These Eight Are Sufficient

**Task-type coverage.** The 23 sub-tasks span both MCQ (factual recall under closed-set constraints) and SAQ (structured extraction under open-ended generation) — the two paradigms used by all cybersecurity LLM evaluation papers in current practice.

**Domain coverage.** The suite covers CTI extraction, vulnerability scoring, root-cause mapping, attack-technique identification, threat-actor attribution, ICS security, general cybersecurity knowledge, and tool proficiency — the full range of sub-domains evaluated by Primus, RedSage, and Foundation-Sec-8B combined.

**Adoption coverage.** Every cybersecurity-specialized LLM that reports multi-benchmark evaluation uses at least three benchmarks from this set. Evaluating on benchmarks that no specialized model uses would measure performance with no comparative baseline.

**Pipeline diversity.** The eight benchmarks represent the full range of pipeline designs in current practice: from benchmarks with no scripts at all (SECURE, SecBench, MMLU-CS) to those with tightly bundled inference+eval scripts (AthenaBench, CyberMetric), covering every combination of prompt availability, evaluation style, and denominator policy.

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

## Hardware

All open-weight and cybersecurity-specialized models are evaluated on a single server node equipped with one **NVIDIA H200 SXM 141 GB GPU**. Models are served via vLLM. Proprietary models (GPT-5.4, Claude Sonnet 4.6) are accessed via their respective APIs and require no local GPU.

## Installation

### Containerized (Docker) Setup

The recommended way to ensure all dependencies (CUDA, PyTorch, vLLM) are correctly installed.

```bash
# Build the image
docker build -t secbench-gpu .

# Run with all GPUs
docker run --gpus all -it secbench-gpu

# Run with a specific GPU (e.g., GPU 0)
docker run --gpus device=0 -it secbench-gpu
```

**Note:** Requires the NVIDIA Container Toolkit — see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html. Reboot if you see GPU reset or detection errors.

### Requirements

- Python ≥ 3.10 (tested with 3.11)
- CUDA-compatible GPU (for local inference)
- 16 GB+ VRAM recommended

### Setup

```bash
git clone https://github.com/yourusername/BenchmarkingSecBenchmarks.git
cd BenchmarkingSecBenchmarks

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

The actively maintained end-to-end implementation lives under `unified-benchmark-pipeline/`.
Paper-oriented reproduction scripts are under `experiment/`.
Legacy regex-baseline scripts are kept under `original-pipeline-exp/` for comparison.

### 1. Collect Model Responses (unified pipeline)

Run inference and save raw responses as JSONL (one file per task):

```bash
cd unified-benchmark-pipeline

# Copy and fill in API credentials for the judge
cp .env.example .env
# edit .env to set AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME

python run_inference_benchmarks.py \
  --model_path "meta-llama/Llama-3.3-70B-Instruct" \
  --use_vllm \
  --tasks mcq rcm vsp ate seceval cybermetric secbench \
  --output_dir outputs/responses_llama33_70b
```

**Common task names (`--tasks`)**
- `mcq`, `seceval`, `cybermetric`, `mmlu-cs`, `secbench`
- `rcm`, `vsp`, `ate`, `cti_taa` (CTI-Bench structured tasks)
- `ckt`, `rms`, `taa`, `athena_ate`, `athena_rcm`, `athena_vsp` (AthenaBench)
- `secure_maet`, `secure_cwet`, `secure_kcv`
- `redsage_frameworks`, `redsage_generals`, `redsage_skills`, `redsage_cli`, `redsage_kali`
- `cissp` (requires `--cissp_path /path/to/cissp.json`)

**Notes**
- `--use_vllm` is optional (local HF inference also works but is slower).
- `--skip_completed` skips tasks that already have a JSONL output file.
- `--n_api_workers` enables parallel collection when using `--use_api`.

### 2. Evaluate with LLM Judge (recommended)

Judging produces a `{output}_detailed/` directory containing `results.json`,
`summary.json`, and `<task>_detailed.jsonl` files.

```bash
cd unified-benchmark-pipeline

python run_evaluate_llm_judge.py \
  --response_dir outputs/responses_llama33_70b \
  --judge_use_api \
  --judge_api_endpoint "$AZURE_ENDPOINT" \
  --judge_api_model "$AZURE_DEPLOYMENT_NAME" \
  --output outputs/judge_llama33_70b/eval_results
```

**Supported judge API styles** (select via `--judge_api_style`):
- `chat_completions` — Azure OpenAI Chat Completions (default)
- `azure_responses` — Azure Responses API
- `anthropic_messages` — Azure-hosted Anthropic /v1/messages (Claude on Azure)

### 3. Post-judge Analysis

After judging, run the re-runnable analysis modules:

```bash
cd unified-benchmark-pipeline/analysis

PYTHONPATH=. python -m analysis.results_table
PYTHONPATH=. python -m analysis.judge_agreement
PYTHONPATH=. python -m analysis.gold_error_voting
```

See `unified-benchmark-pipeline/analysis/README.md` for the full analysis workflow.

### 4. Regex Evaluation Baseline (legacy)

If you need the original regex-style baseline, use the legacy pipeline scripts
under `original-pipeline-exp/`.

```bash
cd original-pipeline-exp

python evaluate.py \
  --model_path "meta-llama/Llama-3.3-70B-Instruct" \
  --tasks mcq rcm vsp ate seceval cybermetric \
  --output eval_results_regex.json
```

### 5. API-based Inference (hosted/commercial models)

```bash
cd unified-benchmark-pipeline

python run_inference_benchmarks.py \
  --use_api \
  --api_style chat_completions \
  --api_endpoint "https://api.openai.com/v1/chat/completions" \
  --api_key "sk-..." \
  --api_model "gpt-4" \
  --tasks seceval cybermetric \
  --output_dir outputs/responses_gpt4
```

**Azure OpenAI:**
```bash
python run_inference_benchmarks.py \
  --use_api \
  --api_style chat_completions \
  --api_endpoint "https://YOUR_RESOURCE.openai.azure.com" \
  --api_key "YOUR_AZURE_KEY" \
  --api_model "YOUR_DEPLOYMENT_NAME" \
  --tasks seceval
```

## Project Structure

```
BenchmarkingSecBenchmarks/
├── requirements.txt                    # Python dependencies
├── unified-benchmark-pipeline/         # Current end-to-end pipeline (recommended)
│   ├── run_inference_benchmarks.py     # Collect responses (JSONL)
│   ├── run_evaluate_llm_judge.py       # LLM-as-judge evaluation
│   ├── analyze_gold_errors.py          # Majority-vote analysis to flag suspect gold answers
│   ├── calibrate_max_tokens.py         # Per-task token-budget calibration via vLLM
│   ├── fix_summaries.py                # Regenerate summary.json from *_detailed.jsonl
│   ├── download_models.sh              # Helper to pre-download model weights
│   ├── run_judge_all_models.sh         # Batch driver: judge across all models
│   ├── .env.example                    # Template for AZURE_API_KEY, AZURE_ENDPOINT
│   ├── experiment/
│   │   └── 01_full_scale_evaluation.py # Full-scale config across all models × tasks
│   └── analysis/                       # Re-runnable post-judge analysis (+ lib/)
│       ├── lib/                        # Shared utilities (loaders, plot_style, voting)
│       ├── results_table.py
│       ├── judge_agreement.py
│       ├── gold_error_voting.py
│       └── …
├── experiment/                         # Paper-oriented scripts (see below)
│   ├── run_all_experiments.py          # Master orchestrator (runs scripts 01–07)
│   ├── 01_full_scale_evaluation.py
│   ├── 02_risys_cluster_analysis.py
│   ├── 03_ate_protocol_analysis.py
│   ├── 04_secure_ceiling_analysis.py
│   ├── 05_backend_variance_analysis.py
│   ├── 06_generate_framework_visualizations.py
│   └── 07_generate_reports.py
├── original-pipeline-exp/              # Legacy regex-baseline scripts
└── Paper/                              # LaTeX paper sources
```

## Running Paper Experiments

For paper-oriented reproduction scripts and visualizations, use `experiment/`.
For the maintained end-to-end pipeline (inference → judge → analysis), use
`unified-benchmark-pipeline/`.

```bash
cd experiment/

# Quick validation with small subset (~1-2 hours, 10 samples per task)
python run_all_experiments.py --subset

# Full-scale evaluation (~5-9 hours, 7 models × 21 tasks)
python run_all_experiments.py

# Skip inference, reuse cached results (runs analysis scripts 2-7 only)
python run_all_experiments.py --skip-inference

# Run only a specific analysis script
python run_all_experiments.py --only 2   # e.g., RISys cluster analysis
```

**Individual scripts:**
| Script | Purpose | Runtime |
|--------|---------|---------|
| `01_full_scale_evaluation.py` | Inference + LLM-judge across all models × tasks | 5-9 hrs |
| `02_risys_cluster_analysis.py` | Circular evaluation evidence (Section 4.1) | <1 min |
| `03_ate_protocol_analysis.py` | ATE format sensitivity bug (Section 4.2) | <1 min |
| `04_secure_ceiling_analysis.py` | SECURE ceiling effect evidence (Section 4.3) | 1-2 hrs |
| `05_backend_variance_analysis.py` | HF vs vLLM backend variance (Section 4.5) | 30 min |
| `06_generate_framework_visualizations.py` | Heatmaps and scatter plots | <1 min |
| `07_generate_reports.py` | Practitioner guides and recommendation matrices | <1 min |

All outputs are written to `experiment/results/`. See `experiment/README.md` for
per-script details, output files, and paper-section mappings.

## Troubleshooting

**Out of memory:** Use fewer tasks, reduce `--max_samples`, or switch to an API judge instead of a local vLLM judge.

**vLLM instability:** Omit `--use_vllm` / `--judge_use_vllm` to fall back to HuggingFace Transformers, or run the judge via API.

**Missing model weights:** Models download automatically on first use (ensure ~500 GB free). To pre-download:
```bash
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')"
```

**Long-running jobs:** Use `screen` or `tmux`, or submit via SLURM:
```bash
screen -S experiments
python run_all_experiments.py
# Detach: Ctrl-A then D  |  Reattach: screen -r experiments
```

## License

MIT License - See [LICENSE](LICENSE) for details
