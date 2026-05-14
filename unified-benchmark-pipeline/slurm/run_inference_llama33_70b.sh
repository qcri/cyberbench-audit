#!/usr/bin/env bash
#SBATCH --job-name=infer_llama33_70b
#SBATCH --partition=gpu-H200
#SBATCH --account=h200
#SBATCH --qos=h200_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:H200_141GB:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=48:00:00
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

# ── Model ─────────────────────────────────────────────────────────────────────
# Llama-3.3-70B-Instruct — download from HuggingFace if not already local.
# bfloat16 70B ≈ 140GB; requires all 4× H200 (564GB total) via tensor parallelism.
MODEL_PATH="$MODELS_DIR/Llama-3.3-70B-Instruct"
MODEL_NAME="Llama-3.3-70B-Instruct"
HF_MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"

CALIB_CEILING=2048
MAX_MODEL_LEN=8192

CALIB_JSON="$ROOT_DIR/slurm/calibration_${MODEL_NAME}.json"
OUTPUT_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"

echo "============================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL_PATH"
echo "Start: $(date)"
echo "============================================================"

# ── PHASE 0 — Download model if not present ───────────────────────────────────
if [ ! -d "$MODEL_PATH" ]; then
    echo ""
    echo "=== PHASE 0: Downloading $HF_MODEL_ID ==="
    mkdir -p "$MODEL_PATH"
    HF_CLI=/export/home/aberriche/miniconda3/envs/seg_zero/bin/huggingface-cli
    if ! "$HF_CLI" download "$HF_MODEL_ID" \
            --local-dir "$MODEL_PATH" \
            --local-dir-use-symlinks False; then
        echo "ERROR: Download failed — likely requires Meta approval (gated model)."
        echo "Request access at https://huggingface.co/$HF_MODEL_ID then re-run."
        rmdir --ignore-fail-on-non-empty "$MODEL_PATH"
        exit 0
    fi
    echo "Download complete: $MODEL_PATH"
else
    echo "Model already present at $MODEL_PATH — skipping download."
fi

# ── PHASE 1 — Token calibration ───────────────────────────────────────────────
echo ""
echo "=== PHASE 1: Token calibration ==="
$PYTHON "$ROOT_DIR/calibrate_max_tokens.py" \
    --model_path    "$MODEL_PATH" \
    --n_samples     10 \
    --ceiling       "$CALIB_CEILING" \
    --gpu_mem       0.90 \
    --max_model_len "$MAX_MODEL_LEN" \
    --output        "$CALIB_JSON"

echo "Calibration complete. Config: $CALIB_JSON"
cat "$CALIB_JSON"

# ── PHASE 2 — Full inference ──────────────────────────────────────────────────
echo ""
echo "=== PHASE 2: Full inference ==="
mkdir -p "$OUTPUT_DIR"

$PYTHON "$ROOT_DIR/run_inference_benchmarks.py" \
    --model_path    "$MODEL_PATH" \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.90 \
    --max_model_len "$MAX_MODEL_LEN" \
    --batch_size    16 \
    --max_tokens_config "$CALIB_JSON" \
    --output_dir    "$OUTPUT_DIR" \
    --tasks \
        mcq rcm vsp ate \
        ckt rms taa athena_ate athena_rcm athena_vsp \
        secure_maet secure_cwet secure_kcv \
        seceval cybermetric mmlu-cs secbench \
        redsage_frameworks redsage_generals redsage_skills \
        redsage_cli redsage_kali

echo ""
echo "============================================================"
echo "Inference complete: $OUTPUT_DIR"
echo "End: $(date)"
echo "============================================================"
