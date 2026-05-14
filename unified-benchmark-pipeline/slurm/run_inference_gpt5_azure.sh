#!/usr/bin/env bash
#SBATCH --job-name=infer_gpt5_azure
#SBATCH --partition=gpu-H200
#SBATCH --account=h200
#SBATCH --qos=h200_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclude=crirdchpxd001,crirdchpxd005
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
MODELS_DIR=/export/cyb-ai-research-data/aberriche/models

mkdir -p "$ROOT_DIR/slurm/logs"
mkdir -p "$ROOT_DIR/outputs"

export TMPDIR=/export/home/aberriche/tmp_slurm/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR=/export/home/aberriche/.cache/triton
export HF_HOME=/export/cyb-ai-research-data/aberriche/hf_cache

# ── API Configuration ─────────────────────────────────────────────────────────
# GPT-5.4 is deployed on Azure using the Responses API.
# Set AZURE_API_KEY in your environment before submitting this job:
#   export AZURE_API_KEY="<your-key>"  && sbatch run_inference_gpt5_azure.sh
# Or embed the key below (not recommended for shared systems).

MODEL_NAME="GPT-5.4"
API_ENDPOINT="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/"
API_MODEL="gpt-5.4"

# Load API key from .env if not already set
if [ -z "${AZURE_API_KEY:-}" ] && [ -f "$ROOT_DIR/.env" ]; then
    set -a; source "$ROOT_DIR/.env"; set +a
fi

if [ -z "${AZURE_API_KEY:-}" ]; then
    echo "ERROR: AZURE_API_KEY not set and not found in .env"
    exit 1
fi

OUTPUT_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"
JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL_NAME}"

# Optional: CISSP dataset path (leave empty to skip CISSP task)
CISSP_PATH=""  # e.g. /export/cyb-ai-research-data/aberriche/data/cissp.json

echo "============================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL_NAME (Azure API)"
echo "API Endpoint: $API_ENDPOINT"
echo "Start: $(date)"
echo "============================================================"

# ── PHASE 1 — Calibration (SKIPPED for API models) ───────────────────────────
echo ""
echo "=== PHASE 1: Token calibration — SKIPPED (API model) ==="

# ── PHASE 2 — Inference via Azure Responses API ───────────────────────────────
echo ""
echo "=== PHASE 2: Full inference (Azure API) ==="
mkdir -p "$OUTPUT_DIR"

TASKS="mcq rcm vsp ate \
       ckt rms taa athena_ate athena_rcm athena_vsp \
       secure_maet secure_cwet secure_kcv \
       seceval cybermetric mmlu-cs secbench \
       redsage_frameworks redsage_generals redsage_skills \
       redsage_cli redsage_kali"

# Add CISSP if path is provided
if [ -n "$CISSP_PATH" ]; then
    TASKS="$TASKS cissp"
fi

CISSP_ARGS=""
if [ -n "$CISSP_PATH" ]; then
    CISSP_ARGS="--cissp_path $CISSP_PATH"
fi

$PYTHON "$ROOT_DIR/run_inference_benchmarks.py" \
    --use_api \
    --api_endpoint  "$API_ENDPOINT" \
    --api_model     "$API_MODEL" \
    --api_key       "$AZURE_API_KEY" \
    --output_dir    "$OUTPUT_DIR" \
    --tasks         $TASKS \
    $CISSP_ARGS

echo ""
echo "============================================================"
echo "Inference complete: $OUTPUT_DIR"
echo "End: $(date)"
echo "============================================================"

# ── PHASE 3 — LLM Judge evaluation ────────────────────────────────────────────
echo ""
echo "=== PHASE 3: LLM Judge evaluation ==="
mkdir -p "$JUDGE_DIR"

$PYTHON "$ROOT_DIR/run_evaluate_llm_judge.py" \
    --response_dir  "$OUTPUT_DIR" \
    --judge_use_api \
    --judge_api_endpoint "$API_ENDPOINT" \
    --judge_api_model    "$API_MODEL" \
    --judge_api_key      "${AZURE_API_KEY}" \
    --n_workers          32 \
    --output             "$JUDGE_DIR/eval_results"

echo "Judge evaluation complete: $JUDGE_DIR"
echo "Done: $(date)"
