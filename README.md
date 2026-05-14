# Benchmarking the Benchmarks: A Meta-Evaluation Framework for Cybersecurity LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


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

The unified pipeline supports 24+ sub-tasks across:

- **CTI-Bench**: MCQ, RCM, VSP, ATE (+ CTI-Bench TAA via TSV subset)
- **AthenaBench**: CKT, RMS (+ expanded extraction subsets and TAA)
- **SECURE**: MAET, CWET, KCV
- **SecEval** · **CyberMetric-500** · **CISSP** · **SecBench** · **MMLU-CS**
- **RedSage-MCQ**: Frameworks, Generals, Skills, CLI, Kali
- **SEvenLLM-Bench**: English subset (structured CTI extraction)

See `unified-benchmark-pipeline/README.md` for the authoritative task list and exact `--tasks` names.

## Installation
### Containerized (Docker) Setup

You can build and run the project in a GPU-enabled Docker container. This is the recommended way to ensure all dependencies (including CUDA, PyTorch, and vLLM) are correctly installed.

#### Build the Docker image:

```bash
docker build -t secbench-gpu .
```

#### Run the container with GPU support:

```bash
# To use all GPUs:
docker run --gpus all -it secbench-gpu

# To use a specific GPU (e.g., GPU 0):
docker run --gpus device=0 -it secbench-gpu
```

**Note:**
- You must have the NVIDIA Container Toolkit installed on your host. See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Your host must have compatible NVIDIA drivers and a CUDA-capable GPU.
- If you see errors about GPU reset or detection, reboot your machine to reset the GPU state.

Once inside the container, you can run all scripts as described below.
### Requirements

- Python ≥ 3.10 (tested with 3.11)
- CUDA-compatible GPU (for local inference)
- 16GB+ VRAM recommended

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/BenchmarkingSecBenchmarks.git
cd BenchmarkingSecBenchmarks

# (Optional) create a virtualenv
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The actively maintained end-to-end implementation lives under `unified-benchmark-pipeline/`.
Legacy scripts are kept under `original-pipeline-exp/` for comparison.

### 1. Collect Model Responses (unified pipeline)

Run inference and save raw responses as JSONL (one file per task):

```bash
cd unified-benchmark-pipeline

python run_inference_benchmarks.py \
  --model_path "meta-llama/Llama-3.1-8B-Instruct" \
  --use_vllm \
  --tasks mcq rcm vsp ate seceval cybermetric cissp \
  --cissp_path ../cissp.json \
  --output_dir outputs/responses_llama31_8b
```

**Common task names (`--tasks`)**
- `mcq`, `seceval`, `cybermetric`, `cissp`, `mmlu-cs`, `secbench`
- `rcm`, `vsp`, `ate` (structured extraction)
- `ckt`, `rms`, `taa`, `cti_taa`, `athena_ate`, `athena_rcm`, `athena_vsp`
- `secure_maet`, `secure_cwet`, `secure_kcv`
- `redsage_frameworks`, `redsage_generals`, `redsage_skills`, `redsage_cli`, `redsage_kali`
- `sevenllm`

**Notes**
- `--use_vllm` is optional (local HF inference also works but is slower).
- `--skip_completed` skips tasks that already have a JSONL output file.
- `--n_api_workers` enables parallel collection when using `--use_api`.

### 2. Evaluate with LLM Judge (recommended)

Judging produces a `{output}_detailed/` directory containing `results.json`,
`summary.json`, and `<task>_detailed.jsonl` files.

```bash
cd unified-benchmark-pipeline

# 1) Copy env template and set judge credentials
cp .env.example .env
# edit .env to set AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME

# 2) Judge previously collected responses
python run_evaluate_llm_judge.py \
  --response_dir outputs/responses_llama31_8b \
  --judge_use_api \
  --judge_api_endpoint "$AZURE_ENDPOINT" \
  --judge_api_model "$AZURE_DEPLOYMENT_NAME" \
  --output outputs/judge_llama31_8b/eval_results
```

Supported judge API styles:
- `chat_completions` (default)
- `azure_responses` (Azure Responses API)
- `anthropic_messages` (Anthropic /v1/messages pass-through)

Select via `--judge_api_style`.

### 3. Regex evaluation baseline (legacy)

If you need the original regex-style baseline, use the legacy pipeline scripts
under `original-pipeline-exp/`.

```bash
cd original-pipeline-exp

python evaluate.py \
  --model_path "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks mcq rcm vsp ate seceval cybermetric cissp \
  --cissp_path ../cissp.json \
  --output eval_results_regex.json
```

### 4. API-based inference (hosted/commercial models)

Evaluate commercial or hosted models:

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

**Azure OpenAI support:**
```bash
cd unified-benchmark-pipeline

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
├── requirements.txt                 # Python dependencies
├── unified-benchmark-pipeline/      # Current end-to-end pipeline (recommended)
│   ├── run_inference_benchmarks.py  # Collect responses (JSONL)
│   ├── run_evaluate_llm_judge.py    # LLM-as-judge evaluation
│   ├── analysis/                    # Re-runnable post-judge analysis (+ lib/)
│   └── experiment/                  # Full-scale experiment driver
├── original-pipeline-exp/           # Legacy scripts kept for comparison
├── experiment/                      # Paper-oriented scripts (legacy; may reference old paths)
└── Paper/                           # LaTeX paper sources
```

## Running Paper Experiments

For paper-oriented reproduction scripts and visualizations, use `experiment/`.
For the maintained end-to-end pipeline (inference → judge → analysis), use
`unified-benchmark-pipeline/`.

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

See `experiment/README.md` for paper-script details, and `unified-benchmark-pipeline/README.md` + `unified-benchmark-pipeline/analysis/README.md` for the maintained pipeline workflows.


## License

MIT License - See [LICENSE](LICENSE) for details
