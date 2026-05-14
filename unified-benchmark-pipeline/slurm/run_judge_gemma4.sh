#!/usr/bin/env bash
#SBATCH --job-name=judge_gemma4
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# Standalone judge re-run for Gemma-4-31B-it responses.
# Uses n_workers=4 (down from 32) to avoid Azure 429 rate-limit throttling.
# Inference responses are already on disk from SLURM job 277434.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks

AZURE_JUDGE_ENDPOINT="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/"
AZURE_JUDGE_MODEL="gpt-5.4"

MODEL_NAME="Gemma-4-31B-it"
OUTPUT_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"
JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL_NAME}"

if [ -z "${AZURE_API_KEY:-}" ] && [ -f "$ROOT_DIR/.env" ]; then
    set -a; source "$ROOT_DIR/.env"; set +a
fi

if [ -z "${AZURE_API_KEY:-}" ]; then
    echo "ERROR: AZURE_API_KEY not set and not found in .env"
    exit 1
fi

mkdir -p "$JUDGE_DIR"

TASKS="mcq rcm seceval secure_cwet secure_kcv secure_maet"

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-local}"
echo "Model responses: $OUTPUT_DIR"
echo "Judge output: $JUDGE_DIR"
echo "n_workers=4  (reduced from 32 to avoid 429 throttling)"
echo "Start: $(date)"
echo "============================================================"

$PYTHON "$ROOT_DIR/run_evaluate_llm_judge.py" \
    --response_dir       "$OUTPUT_DIR" \
    --judge_use_api \
    --judge_api_endpoint "$AZURE_JUDGE_ENDPOINT" \
    --judge_api_model    "$AZURE_JUDGE_MODEL" \
    --judge_api_key      "${AZURE_API_KEY}" \
    --n_workers          4 \
    --tasks              $TASKS \
    --output             "$JUDGE_DIR/eval_results"

echo "============================================================"
echo "Judge evaluation complete: $JUDGE_DIR"
echo "Done: $(date)"
echo "============================================================"
