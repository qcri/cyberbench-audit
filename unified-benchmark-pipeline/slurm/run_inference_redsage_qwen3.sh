#!/usr/bin/env bash
#SBATCH --job-name=infer_redsage_qwen3
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

mkdir -p "$ROOT_DIR/slurm/logs"
mkdir -p "$ROOT_DIR/outputs"

# ── Temp dir fix: /var/tmp and /tmp may be full on compute nodes ──────────────
# Triton JIT and other tools write to TMPDIR; redirect to writable home path.
export TMPDIR=/export/home/aberriche/tmp_slurm/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR=/export/home/aberriche/.cache/triton
export HF_HOME=/export/cyb-ai-research-data/aberriche/hf_cache

MODELS_DIR=/export/cyb-ai-research-data/aberriche/models

AZURE_JUDGE_ENDPOINT="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/openai/responses?api-version=2025-04-01-preview"
AZURE_JUDGE_MODEL="gpt-4.1"

# ── Model ─────────────────────────────────────────────────────────────────────
# RedSage-Qwen3-8B-DPO is a THINKING model (Qwen3 backbone).
# It produces <think>…</think> blocks before the final answer.
# max_model_len is set higher than default to accommodate long thinking chains.
MODEL_PATH="$MODELS_DIR/RedSage-Qwen3-8B-DPO"
MODEL_NAME="RedSage-Qwen3-8B-DPO"

# Fallback to HF ID if local path not available
if [ ! -d "$MODEL_PATH" ]; then
    MODEL_PATH="RISys-Lab/RedSage-Qwen3-8B-DPO"
fi

# Calibration ceiling — raise to 8192 if you see truncation warnings
CALIB_CEILING=4096
# vLLM context window — Qwen3 supports 32k; 16k is enough for most tasks
MAX_MODEL_LEN=16384

CALIB_JSON="$ROOT_DIR/slurm/calibration_${MODEL_NAME}.json"
OUTPUT_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"
JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL_NAME}"

echo "============================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL_PATH"
echo "Start: $(date)"
echo "============================================================"

# ── PHASE 1 — Token calibration (10 samples × 21 tasks, ~20–40 min) ──────────
echo ""
echo "=== PHASE 1: Token calibration ==="
$PYTHON "$ROOT_DIR/calibrate_max_tokens.py" \
    --model_path "$MODEL_PATH" \
    --n_samples  10 \
    --ceiling    "$CALIB_CEILING" \
    --gpu_mem    0.85 \
    --max_model_len "$MAX_MODEL_LEN" \
    --output     "$CALIB_JSON"

echo "Calibration complete.  Config: $CALIB_JSON"
cat "$CALIB_JSON"

# ── PHASE 2 — Full inference (all 21 tasks) ───────────────────────────────────
echo ""
echo "=== PHASE 2: Full inference ==="
mkdir -p "$OUTPUT_DIR"

$PYTHON "$ROOT_DIR/run_inference_benchmarks.py" \
    --model_path    "$MODEL_PATH" \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.85 \
    --max_model_len "$MAX_MODEL_LEN" \
    --batch_size    32 \
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

# ── PHASE 3 — LLM Judge evaluation ────────────────────────────────────────────
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
