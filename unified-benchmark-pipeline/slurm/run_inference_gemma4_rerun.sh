#!/usr/bin/env bash
#SBATCH --job-name=gemma4_rerun
#SBATCH --partition=gpu-H200
#SBATCH --account=h200
#SBATCH --qos=h200_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:H200_141GB:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --exclude=crirdchpxd001,crirdchpxd005
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# Re-run only the 6 tasks with high empty-response rates from the first Gemma-4 run.
# Root cause: KV cache limited to 512 blocks (num_gpu_blocks_override) due to
# heterogeneous head dims; max_model_len=4096 left only 2 concurrent slots.
# Fix: max_model_len=2048 → 4 slots fit (8192 KV tokens / 2048 = 4).
# Max tokens ceiling also set to 1024 to leave headroom within the 2048 context.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
MODELS_DIR=/export/cyb-ai-research-data/aberriche/models

AZURE_JUDGE_ENDPOINT="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/"
AZURE_JUDGE_MODEL="gpt-5.4"

mkdir -p "$ROOT_DIR/slurm/logs"

if [ -z "${AZURE_API_KEY:-}" ] && [ -f "$ROOT_DIR/.env" ]; then
    set -a; source "$ROOT_DIR/.env"; set +a
fi

export TMPDIR=/export/home/aberriche/tmp_slurm/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR=/export/home/aberriche/.cache/triton
export HF_HOME=/export/cyb-ai-research-data/aberriche/hf_cache

MODEL_PATH="$MODELS_DIR/gemma-4-31B"
MODEL_NAME="Gemma-4-31B-it"
if [ ! -d "$MODEL_PATH" ]; then
    MODEL_PATH="google/gemma-4-31B"
fi

MAX_MODEL_LEN=2048
OUTPUT_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"
JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL_NAME}"

echo "============================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL_PATH  max_model_len=$MAX_MODEL_LEN"
echo "Re-running 6 tasks with high empty-response rates"
echo "Start: $(date)"
echo "============================================================"

# Overwrite only the 6 affected tasks (others already good in $OUTPUT_DIR)
TASKS="mcq rcm seceval secure_cwet secure_kcv secure_maet"

mkdir -p "$OUTPUT_DIR"

$PYTHON "$ROOT_DIR/run_inference_benchmarks.py" \
    --model_path    "$MODEL_PATH" \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.90 \
    --max_model_len "$MAX_MODEL_LEN" \
    --batch_size    32 \
    --output_dir    "$OUTPUT_DIR" \
    --tasks         $TASKS

echo ""
echo "============================================================"
echo "Re-inference complete: $OUTPUT_DIR"
echo "End: $(date)"
echo "============================================================"

echo ""
echo "=== Re-running LLM Judge for affected tasks ==="
if [ -z "${AZURE_API_KEY:-}" ]; then
    echo "AZURE_API_KEY not set — skipping judge"
    exit 0
fi
mkdir -p "$JUDGE_DIR"

$PYTHON "$ROOT_DIR/run_evaluate_llm_judge.py" \
    --response_dir  "$OUTPUT_DIR" \
    --judge_use_api \
    --judge_api_endpoint "$AZURE_JUDGE_ENDPOINT" \
    --judge_api_model    "$AZURE_JUDGE_MODEL" \
    --judge_api_key      "${AZURE_API_KEY:-}" \
    --n_workers          32 \
    --tasks             $TASKS \
    --output             "$JUDGE_DIR/eval_results"

echo "Judge re-evaluation complete: $JUDGE_DIR"
echo "Done: $(date)"
