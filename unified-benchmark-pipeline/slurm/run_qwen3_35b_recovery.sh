#!/usr/bin/env bash
#SBATCH --job-name=qwen3_recovery
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

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
MODELS_DIR=/export/cyb-ai-research-data/aberriche/models

AZURE_JUDGE_ENDPOINT="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/openai/responses?api-version=2025-04-01-preview"
AZURE_JUDGE_MODEL="gpt-4.1"

mkdir -p "$ROOT_DIR/slurm/logs"
export TMPDIR=/export/home/aberriche/tmp_slurm/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR=/export/home/aberriche/.cache/triton
export HF_HOME=/export/cyb-ai-research-data/aberriche/hf_cache

MODEL_PATH="$MODELS_DIR/Qwen3.6-35B-A3B"
if [ ! -d "$MODEL_PATH" ]; then
    MODEL_PATH="Qwen/Qwen3.6-35B-A3B"
fi
MODEL_NAME="Qwen3.6-35B-A3B"

CALIB_JSON="$ROOT_DIR/slurm/calibration_${MODEL_NAME}.json"
OUTPUT_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"
JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL_NAME}"

echo "============================================================"
echo "Recovery Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL_PATH"
echo "Start: $(date)"
echo "Recovering 13 failed tasks (Phase 1 skipped — using existing calibration)"
echo "============================================================"

# Skip Phase 1 — calibration JSON already exists
if [ ! -f "$CALIB_JSON" ]; then
    echo "ERROR: Calibration JSON not found at $CALIB_JSON"
    exit 1
fi
echo "Using existing calibration: $CALIB_JSON"

# ── PHASE 2 — Re-run only the 13 failed tasks ─────────────────────────────────
echo ""
echo "=== PHASE 2: Recovery inference (13 failed tasks) ==="
mkdir -p "$OUTPUT_DIR"

# These 13 tasks failed in the original run due to EngineCore timeout during athena_vsp
# athena_vsp max_tokens patched to 4096 in calibration JSON to avoid timeout
TASKS="athena_vsp \
       secure_maet secure_cwet secure_kcv \
       seceval cybermetric mmlu-cs secbench \
       redsage_frameworks redsage_generals redsage_skills \
       redsage_cli redsage_kali"

$PYTHON "$ROOT_DIR/run_inference_benchmarks.py" \
    --model_path    "$MODEL_PATH" \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.85 \
    --batch_size    32 \
    --max_tokens_config "$CALIB_JSON" \
    --output_dir    "$OUTPUT_DIR" \
    --tasks         $TASKS

echo ""
echo "============================================================"
echo "Recovery inference complete: $OUTPUT_DIR"
echo "End: $(date)"
echo "============================================================"

# ── PHASE 3 — LLM Judge (skip if no API key) ──────────────────────────────────
echo ""
echo "=== PHASE 3: LLM Judge evaluation ==="
if [ -z "${AZURE_API_KEY:-}" ]; then
    echo "AZURE_API_KEY not set — skipping Phase 3 (re-run with key when available)"
    echo "Responses saved to: $OUTPUT_DIR"
    echo "Done: $(date)"
    exit 0
fi
mkdir -p "$JUDGE_DIR"

$PYTHON "$ROOT_DIR/run_evaluate_llm_judge.py" \
    --response_dir  "$OUTPUT_DIR" \
    --judge_use_api \
    --judge_api_endpoint "$AZURE_JUDGE_ENDPOINT" \
    --judge_api_model    "$AZURE_JUDGE_MODEL" \
    --judge_api_key      "${AZURE_API_KEY:-}" \
    --judge_api_style    azure_responses \
    --output        "$JUDGE_DIR/eval_results"

echo "Judge evaluation complete: $JUDGE_DIR"
echo "Done: $(date)"
