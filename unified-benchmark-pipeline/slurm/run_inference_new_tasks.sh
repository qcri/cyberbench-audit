#!/usr/bin/env bash
#SBATCH --job-name=infer_new_tasks
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

# Run cti_taa + sevenllm inference for one model, appending to its existing
# outputs/responses_<MODEL>/ directory. --skip_completed protects existing tasks.
#
# Usage:
#   sbatch --export=ALL,MODEL_NAME=<name>,MODEL_PATH=<path>[,MAX_SAMPLES=<n>] \
#          run_inference_new_tasks.sh
#
# MODEL_NAME — output dir suffix (becomes outputs/responses_<MODEL_NAME>/).
# MODEL_PATH — local path or HF repo id.
# MAX_SAMPLES (optional) — cap per task (smoke-test mode).
# TASKS (optional) — override task list (default: cti_taa sevenllm).

if [ -z "${MODEL_NAME:-}" ] || [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: set MODEL_NAME and MODEL_PATH via --export"
    exit 1
fi

source /export/home/aberriche/miniconda3/etc/profile.d/conda.sh
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
MODELS_DIR=/export/cyb-ai-research-data/aberriche/models

mkdir -p "$ROOT_DIR/slurm/logs"
mkdir -p "$ROOT_DIR/outputs"

export TMPDIR=/export/home/aberriche/tmp_slurm/${SLURM_JOB_ID:-local}
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR=/export/home/aberriche/.cache/triton
export HF_HOME=/export/cyb-ai-research-data/aberriche/hf_cache

TASKS="${TASKS:-cti_taa sevenllm}"
CALIB_CEILING=2048
MAX_MODEL_LEN=8192

CALIB_JSON="$ROOT_DIR/slurm/calibration_${MODEL_NAME}.json"
OUTPUT_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"
mkdir -p "$OUTPUT_DIR"

MAX_SAMPLES_ARG=""
if [ -n "${MAX_SAMPLES:-}" ]; then
    MAX_SAMPLES_ARG="--max_samples ${MAX_SAMPLES}"
fi

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-local}"
echo "Model: $MODEL_PATH ($MODEL_NAME)"
echo "Tasks: $TASKS"
echo "Output: $OUTPUT_DIR"
echo "Max samples: ${MAX_SAMPLES:-none (full dataset)}"
echo "Start: $(date)"
echo "============================================================"

# ── Calibration: only re-run if existing calibration is missing.
# (cti_taa/sevenllm fall back to script defaults if not in the calibration JSON,
# so calibration is optional but recommended for token-budget consistency.)
if [ ! -f "$CALIB_JSON" ]; then
    echo ""
    echo "=== Calibration (no existing config) ==="
    $PYTHON "$ROOT_DIR/calibrate_max_tokens.py" \
        --model_path    "$MODEL_PATH" \
        --n_samples     10 \
        --ceiling       "$CALIB_CEILING" \
        --gpu_mem       0.90 \
        --max_model_len "$MAX_MODEL_LEN" \
        --output        "$CALIB_JSON"
else
    echo "Reusing existing calibration: $CALIB_JSON"
fi

# ── Inference for the new tasks only ─────────────────────────────────────────
echo ""
echo "=== Inference for new tasks ==="
$PYTHON "$ROOT_DIR/run_inference_benchmarks.py" \
    --model_path    "$MODEL_PATH" \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.90 \
    --max_model_len "$MAX_MODEL_LEN" \
    --batch_size    32 \
    --max_tokens_config "$CALIB_JSON" \
    --output_dir    "$OUTPUT_DIR" \
    --tasks         $TASKS \
    --skip_completed \
    $MAX_SAMPLES_ARG

echo ""
echo "============================================================"
echo "Done: $OUTPUT_DIR"
echo "End: $(date)"
echo "============================================================"
