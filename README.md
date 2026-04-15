# Benchmarking the Benchmarks: A Meta-Evaluation Framework for Cybersecurity LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A systematic meta-evaluation of cybersecurity LLM benchmarks with unified evaluation harness and reproducible scoring protocols.**

## Overview

The cybersecurity LLM evaluation ecosystem has expanded rapidly, yielding **12 benchmark families** and **5 security-specialized models** between 2023-2025—yet no systematic analysis of their quality, coverage, or mutual consistency exists. This repository provides:

- **Two-axis taxonomy** separating knowledge benchmarks (factual recall, CTI reasoning, structured extraction) from capability benchmarks (adversarial robustness, offensive operations)
- **Six-dimension meta-evaluation framework** (D1-D6) for assessing benchmark quality
- **Unified evaluation harness** supporting 20+ cybersecurity tasks across 9 benchmark families
- **Reproducible benchmarking protocol** addressing seven critical gaps in current evaluation practices

### Key Findings

1. **Circular evaluation cluster**: CTI-Bench, AthenaBench, SECURE, and RedSage model all originate from the same research group
2. **Protocol bugs**: ATT&CK Technique Extraction (ATE) task scores 0% F₁ universally due to regex evaluation discarding semantically correct responses
3. **Ceiling effects**: Three SECURE tasks saturate at 100% accuracy with base 8B instruction-tuned models
4. **Backend sensitivity**: Evaluation backend choice (HuggingFace vs. vLLM) changes accuracy from 57.89% to 47.50% on identical responses
5. **Multilingual gap**: Arabic is entirely absent from all 12 benchmarks and all 5 security LLMs
6. **Harness fragmentation**: Three incompatible evaluation harnesses prevent cross-paper model comparisons

## Meta-Evaluation Framework

Our framework assesses benchmarks across six dimensions:

| Dimension | Criterion | Scoring |
|-----------|-----------|---------|
| **D1: Scale & Balance** | Task size (items per task) | **H**: ≥500, **M**: 100-499, **L**: <100, **N**: Undisclosed |
| **D2: Validation Provenance** | Label quality and source | **H**: Expert-validated/Contest, **M**: Partial review, **L**: Synthetic, **N**: Undisclosed |
| **D3: Task Diversity** | Coverage of paradigms | **H**: 3+ paradigms, **M**: 2 paradigms, **L**: 1 paradigm |
| **D4: Evaluation Protocol** | Robustness of metrics | **H**: LLM-judge, **M**: Mixed, **L**: Regex-only, **N**: Undisclosed |
| **D5: Reproducibility** | Code and data availability | **H**: Full release, **M**: Partial, **L**: Limited, **N**: None |
| **D6: Multilingual** | Language coverage | **H**: 3+ families, **M**: 2 languages, **L**: 1 non-English, **N**: English-only |

## Supported Benchmarks

### Knowledge Benchmarks (21 Tasks)

#### CTI-Bench (RISys-Lab)
- **MCQ**: Multiple-choice cybersecurity knowledge
- **RCM**: CWE (Common Weakness Enumeration) identification
- **VSP**: CVSS vector scoring and prediction
- **ATE**: ATT&CK technique extraction
- **TAA**: Threat actor attribution

#### AthenaBench (GitHub JSONL)
- **CKT**: Cybersecurity knowledge testing
- **RMS**: Mitigation strategy identification

#### SECURE
- **MAET**: Multi-aspect evaluation task
- **CWET**: CWE enumeration task
- **KCV**: Knowledge comprehension and verification

#### General Cybersecurity Benchmarks
- **SecEval**: 2,126 cybersecurity knowledge questions (multi-select MCQ)
- **CyberMetric-500**: Expert-curated cybersecurity questions
- **CISSP**: Certification exam questions
- **SecBench**: Large-scale security benchmark (44,823 items)
- **MMLU-CS**: Computer security subset of MMLU
- **RedSage-Bench**: 5 categories (Frameworks, Generals, Skills, CLI, Kali)

## Installation

### Requirements

- Python ≥ 3.10 (tested with 3.11)
- CUDA-compatible GPU (for local inference)
- 16GB+ VRAM recommended

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/BenchmarkingSecBenchmarks.git
cd BenchmarkingSecBenchmarks

# Install dependencies
pip install -r requirements.txt

# Optional: Install vLLM for fast batched inference
pip install vllm
```

### Dependencies

Core packages:
- `torch>=2.0.0` - PyTorch deep learning framework
- `transformers>=4.35.0` - HuggingFace model library
- `peft>=0.7.0` - Parameter-efficient fine-tuning (optional)
- `datasets>=2.14.0` - HuggingFace datasets
- `numpy`, `scikit-learn` - Metrics and evaluation
- `tqdm`, `requests`, `pyyaml`, `python-dotenv` - Utilities

## Usage

### 1. Collect Model Responses

Run inference on benchmarks and save responses as JSONL:

```bash
python run_inference_benchmarks.py \
  --model_path "meta-llama/Llama-3.1-8B-Instruct" \
  --benchmarks ctibench_mcq seceval cybermetric cissp \
  --output_dir ./outputs \
  --temperature 0.1 \
  --max_tokens 1024
```

**Supported benchmark flags:**
- `ctibench_mcq`, `ctibench_rcm`, `ctibench_vsp`, `ctibench_ate`, `ctibench_taa`
- `athenabench_ckt`, `athenabench_rms`
- `secure_maet`, `secure_cwet`, `secure_kcv`
- `seceval`, `cybermetric`, `cissp`, `secbench`, `mmlu_cs`
- `redsage_frameworks`, `redsage_generals`, `redsage_skills`, `redsage_cli`, `redsage_kali`

**Optional arguments:**
- `--base_model`: Base model name (optional for some models)
- `--is_base`: Load as base model without adapters
- `--use_vllm`: Use vLLM for fast batched inference
- `--vllm_gpu_memory_utilization`: GPU memory utilization (default: 0.9)

### 2. Evaluate with Regex (Original Protocol)

```bash
python evaluate.py \
  --model_path "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks mcq rcm vsp ate cybermetric seceval cissp \
  --output eval_results.json
```

**Note**: This uses regex-based extraction and is provided for baseline comparison. Known to fail on structured extraction tasks (ATE, RCM, RMS).

**Optional arguments:**
- `--use_api`: Use API endpoint instead of local model
- `--max_samples`: Limit samples per task (for testing)
- `--cissp_path`: Path to CISSP dataset JSON file

### 3. Evaluate with LLM Judge (Recommended)

Robust evaluation using LLM-as-a-Judge:

```bash
python run_evaluate_llm_judge.py \
  --responses_dir ./outputs \
  --judge_model "meta-llama/Llama-3.1-8B-Instruct" \
  --output_file results_llm_judge.json \
  --temperature 0.0
```

**LLM Judge advantages:**
- Handles prose responses with embedded IDs (e.g., "The technique is T1566 (Phishing)")
- Semantic matching for equivalent answers
- Normalizes CWE/MITRE ID formats
- Prevents systematic 0% F₁ failures

**Optional arguments:**
- `--use_api`: Use API endpoint instead of local model
- `--api_endpoint`: OpenAI-compatible API endpoint
- `--api_key`: API authentication key
- `--judge_backend`: `transformers` or `vllm`

### 4. API-based Evaluation

Evaluate commercial or hosted models:

```bash
python run_inference_benchmarks.py \
  --use_api \
  --api_endpoint "https://api.openai.com/v1/chat/completions" \
  --api_key "sk-..." \
  --model_name "gpt-4" \
  --benchmarks ctibench_mcq seceval \
  --output_dir ./outputs_api
```

**Azure OpenAI support:**
```bash
python run_inference_benchmarks.py \
  --use_api \
  --api_endpoint "https://YOUR_RESOURCE.openai.azure.com" \
  --api_key "YOUR_AZURE_KEY" \
  --model_name "YOUR_DEPLOYMENT_NAME" \
  --api_version "2024-02-15-preview" \
  --benchmarks seceval
```

### 5. Flag Incorrect Benchmark Answers

After running LLM judge evaluation on multiple models, identify questions with potentially incorrect key answers using model agreement voting:

```bash
python flag_wrong_key_answers.py \
  --detailed_results_dirs eval_llm_judge_model1_detailed eval_llm_judge_model2_detailed \
  --agreement_threshold 0.5 \
  --judge_model "meta-llama/Llama-3.1-8B-Instruct" \
  --output flagged_questions_report.json
```

**With vLLM for faster inference:**
```bash
python flag_wrong_key_answers.py \
  --detailed_results_dirs eval_llm_judge_model1_detailed eval_llm_judge_model2_detailed \
  --agreement_threshold 0.5 \
  --judge_model "meta-llama/Llama-3.1-8B-Instruct" \
  --judge_use_vllm \
  --judge_gpu_memory_utilization 0.8 \
  --output flagged_questions_report.json
```

This script analyzes evaluation results from multiple models and flags questions where most models agree on an alternative answer to the benchmark's key answer. This helps identify potential errors in benchmark ground truth labels.

**Voting Mechanism:**
- If ≥ threshold fraction of models agree on answer x, but the key answer is y (x ≠ y), then the question is flagged as having a likely wrong key answer
- For MCQ tasks: Uses exact matching after normalization
- For open-ended tasks: Uses LLM judge for semantic comparison (required)

**Arguments:**
- `--detailed_results_dirs`: Directories containing *_detailed.jsonl files from run_evaluate_llm_judge.py
- `--agreement_threshold`: Fraction of models that must agree on alternative answer (default: 0.5)
- `--judge_model`: Judge model path for semantic comparison of open-ended answers (required if open-ended tasks are present)
- `--judge_use_vllm`: Use vLLM for fast judge inference (requires vllm package)
- `--judge_gpu_memory_utilization`: GPU memory fraction to use with vLLM judge (0.0-1.0, default: 0.9)
- `--output`: Output JSON file with flagging report
- `--update_jsonl`: Update original JSONL files with wrong_key_answer property

## Project Structure

```
BenchmarkingSecBenchmarks/
├── evaluate.py                      # Regex-based evaluation (baseline)
├── run_inference_benchmarks.py      # Collect model responses (JSONL)
├── run_evaluate_llm_judge.py        # LLM-judge evaluation (recommended)
├── flag_wrong_key_answers.py        # Flag incorrect benchmark answers
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore patterns
├── outputs/                         # Saved model responses (JSONL)
└── experiment/                      # Paper experiments & analysis
    ├── run_all_experiments.py       # Master orchestrator (runs all 7 scripts)
    ├── 01_full_scale_evaluation.py  # Main evaluation (7 models × 21 tasks)
    ├── 02_risys_cluster_analysis.py # Circular evaluation analysis
    ├── 03_ate_protocol_analysis.py  # ATE format sensitivity bug
    ├── 04_secure_ceiling_analysis.py # SECURE ceiling effects
    ├── 05_backend_variance_analysis.py # HF vs vLLM comparison
    ├── 06_generate_framework_visualizations.py # Create plots
    ├── 07_generate_reports.py       # Practitioner guides
    └── README.md                     # Detailed experiment documentation
```

## Running Paper Experiments

For reproducing paper results (Section 5 - Empirical Evaluation):

```bash
cd experiment/

# Quick validation with small subset (1-2 hours)
python run_all_experiments.py --subset

# Full-scale evaluation (5-9 hours, 7 models × 21 tasks)
python run_all_experiments.py
```

This runs all 7 experiment scripts and generates:
- Full evaluation results with confidence intervals
- Gap analysis evidence (circular eval, protocol bugs, ceiling effects)
- Framework visualizations (heatmaps, scatter plots)
- Practitioner guides and recommendations

See [experiment/README.md](experiment/README.md) for detailed documentation.

## Evaluated Models

### Base Model
- **Llama-3.1-8B-Instruct**: General-purpose instruction-tuned model

### Security-Specialized LLMs
- **Llama-Primus-Merged** (TrendMicro): CPT with 2.75B security tokens + 835 reasoning samples
- **Llama-Primus-Base** (TrendMicro): CPT with 2.75B security tokens
- **Foundation-Sec-8B-Instruct**: CPT with 5.1B security tokens + 28K SFT samples
- **RedSage-8B-Ins** (RISys-Lab): CPT with 11.8B security tokens + 266K SFT samples
- **RedSage-8B-DPO** (RISys-Lab): RedSage-Ins + DPO alignment

### Multilingual
- **QCRI/Fanar-1-9B-Instruct**: Arabic-capable model for multilingual evaluation

## Evaluation Protocol Recommendations

Based on our gap analysis, we propose five standard requirements:

1. **Unified Harness**: Use shared evaluation infrastructure with pinned dependencies
2. **LLM-Judge for Structured Tasks**: Replace regex with LLM-based extraction for CWE/MITRE IDs
3. **Backend Consistency**: Document inference backend (transformers/vLLM/API) and verify stability
4. **Provenance Disclosure**: Report benchmark authorship clusters and potential circular dependencies
5. **Multilingual Coverage**: Include non-English evaluation (minimum: Arabic for cybersecurity)


## License

MIT License - See [LICENSE](LICENSE) for details
