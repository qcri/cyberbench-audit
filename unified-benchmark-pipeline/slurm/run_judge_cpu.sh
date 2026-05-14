#!/usr/bin/env bash
#SBATCH --job-name=judge_gpt54
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# Usage: sbatch --export=ALL,MODEL_NAME=Llama-Primus-Merged run_judge_cpu.sh
# MODEL_NAME must be set via --export or environment before sbatch

if [ -z "${MODEL_NAME:-}" ]; then
    echo "ERROR: MODEL_NAME is not set. Use: sbatch --export=ALL,MODEL_NAME=<name> $0"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks

mkdir -p "$ROOT_DIR/slurm/logs"

# Load API key from .env
if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
fi

if [ -z "${AZURE_API_KEY:-}" ]; then
    echo "ERROR: AZURE_API_KEY not set and not found in .env"
    exit 1
fi

AZURE_JUDGE_ENDPOINT="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/"
AZURE_JUDGE_MODEL="gpt-5.4"
N_WORKERS=8

RESPONSE_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"
JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL_NAME}"

if [ ! -d "$RESPONSE_DIR" ]; then
    echo "ERROR: Response dir not found: $RESPONSE_DIR"
    exit 1
fi

N_FILES=$(ls "$RESPONSE_DIR"/*.jsonl 2>/dev/null | wc -l)
echo "============================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL_NAME ($N_FILES task files)"
echo "Judge: $AZURE_JUDGE_MODEL at $AZURE_JUDGE_ENDPOINT"
echo "Workers: $N_WORKERS"
echo "Start: $(date)"
echo "============================================================"

mkdir -p "$JUDGE_DIR"

$PYTHON "$ROOT_DIR/run_evaluate_llm_judge.py" \
    --response_dir  "$RESPONSE_DIR" \
    --judge_use_api \
    --judge_api_endpoint "$AZURE_JUDGE_ENDPOINT" \
    --judge_api_model    "$AZURE_JUDGE_MODEL" \
    --judge_api_key      "${AZURE_API_KEY}" \
    --n_workers          "$N_WORKERS" \
    --output             "$JUDGE_DIR/eval_results"

echo ""
echo "============================================================"
echo "Judge complete: $JUDGE_DIR"
echo "End: $(date)"
echo "============================================================"
