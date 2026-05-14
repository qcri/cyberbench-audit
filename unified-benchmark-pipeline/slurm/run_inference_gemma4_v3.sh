#!/usr/bin/env bash
#SBATCH --job-name=gemma4_v3
#SBATCH --partition=gpu-H200
#SBATCH --account=h200
#SBATCH --qos=h200_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:H200_141GB:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --exclude=crirdchpxd001,crirdchpxd005
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# Gemma-4-31B v3 re-run — fixes for two separate empty-response causes:
#
# Root cause 1 — KV profiler returns 0 blocks:
#   Gemma-4 has heterogeneous KV heads (sliding: 16 heads × head_dim=256;
#   global: 4 heads × global_head_dim=512). vLLM's block allocator cannot
#   compute a unified block size → num_gpu_blocks=0. Fix:
#     VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1  (as suggested by vLLM itself;
#       confirmed in v3 run: profiler correctly computes 172,348 blocks now)
#     --enforce_eager  (disables CUDA graph, stabilises execution)
#     NOTE: --num_gpu_blocks_override is NOT used here. v3 showed that with the
#       env var fix the profiler returns 172,348 (correct). The override had been
#       set to 2048, which constrained KV to 5,456 tokens (0.36x concurrency).
#       Removing it lets vLLM use the full 172k block budget.
#
# Root cause 2 — Prompt tokens exceed max_model_len:
#   KCV prompts contain raw CVE JSON that tokenizes at ~1 char/token.
#   Median KCV prompt is 3,204 chars ≈ 3,200 tokens — exceeds previous
#   max_model_len=2048 and even max_model_len=4096 for longer CVE records.
#   Fix: max_model_len=16384 covers all observed prompts (max 11,290 chars).
#   Reduced batch_size=8 to limit concurrent KV pressure at this context size.

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

# Fix for KV profiler returning 0 blocks (see comments above)
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

MODEL_PATH="$MODELS_DIR/gemma-4-31B"
MODEL_NAME="Gemma-4-31B-it"
if [ ! -d "$MODEL_PATH" ]; then
    MODEL_PATH="google/gemma-4-31B"
fi

MAX_MODEL_LEN=16384
OUTPUT_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"
JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL_NAME}"

# v3 run confirmed that VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 + enforce_eager
# fixes the profiler: it now correctly computes 172,348 blocks (vs 0 before).
# The --num_gpu_blocks_override has been removed so vLLM uses that full budget.
# With 172k blocks and max_model_len=16384: ~168 concurrent sequences → fast.

echo "============================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL_PATH"
echo "max_model_len=$MAX_MODEL_LEN  (no block override — profiler now correct)"
echo "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=$VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"
echo "Start: $(date)"
echo "============================================================"

# Re-run the 6 tasks that had high empty-response rates.
TASKS="mcq rcm seceval secure_cwet secure_kcv secure_maet"

mkdir -p "$OUTPUT_DIR"

$PYTHON "$ROOT_DIR/run_inference_benchmarks.py" \
    --model_path                "$MODEL_PATH" \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.90 \
    --max_model_len             "$MAX_MODEL_LEN" \
    --enforce_eager \
    --batch_size                32 \
    --output_dir                "$OUTPUT_DIR" \
    --tasks                     $TASKS

echo ""
echo "============================================================"
echo "Inference complete: $OUTPUT_DIR"
echo "End: $(date)"
echo "============================================================"

echo ""
echo "=== Running LLM Judge for re-run tasks ==="
if [ -z "${AZURE_API_KEY:-}" ]; then
    echo "AZURE_API_KEY not set — skipping judge"
    exit 0
fi
mkdir -p "$JUDGE_DIR"

$PYTHON "$ROOT_DIR/run_evaluate_llm_judge.py" \
    --response_dir       "$OUTPUT_DIR" \
    --judge_use_api \
    --judge_api_endpoint "$AZURE_JUDGE_ENDPOINT" \
    --judge_api_model    "$AZURE_JUDGE_MODEL" \
    --judge_api_key      "${AZURE_API_KEY:-}" \
    --n_workers          32 \
    --tasks              $TASKS \
    --output             "$JUDGE_DIR/eval_results"

echo "Judge evaluation complete: $JUDGE_DIR"
echo "Done: $(date)"
