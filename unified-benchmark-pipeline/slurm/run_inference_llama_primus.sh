#!/usr/bin/env bash
#SBATCH --job-name=infer_llama_primus
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
# Llama-Primus-Merged (TrendMicro PRIMUS paper) — Llama-3.1-8B backbone,
# CPT + SFT fine-tuned on 23K cybersecurity samples.  Not a thinking model.
#
# ⚠ IMPORTANT: Confirm the correct path with the team.
#   Options:
#     HuggingFace: TrendMicro-AI/LLM-Primus-8B-Merged  (verify on HF Hub)
#     Local path:  /path/to/local/Llama-Primus-Merged
#   Update MODEL_PATH below accordingly.
MODEL_PATH="${LLAMA_PRIMUS_PATH:-$MODELS_DIR/Llama-Primus-Merged}"
MODEL_NAME="Llama-Primus-Merged"

# Fallback to HF ID if local path not available
if [ ! -d "$MODEL_PATH" ]; then
    MODEL_PATH="trendmicro-ailab/Llama-Primus-Merged"
fi

CALIB_CEILING=2048
MAX_MODEL_LEN=8192

CALIB_JSON="$ROOT_DIR/slurm/calibration_${MODEL_NAME}.json"
OUTPUT_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"
JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL_NAME}"

echo "============================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model: $MODEL_PATH"
echo "Start: $(date)"
echo "============================================================"

# ── Verify model path ─────────────────────────────────────────────────────────
if [[ "$MODEL_PATH" == "trendmicro-ailab/Llama-Primus-Merged" ]]; then
    echo "⚠  WARNING: Using default MODEL_PATH=$MODEL_PATH"
    echo "   If this is wrong, set LLAMA_PRIMUS_PATH before sbatch:"
    echo "   LLAMA_PRIMUS_PATH=/path/to/model sbatch slurm/run_inference_llama_primus.sh"
    echo ""
fi

# ── PHASE 1 — Token calibration ───────────────────────────────────────────────
echo ""
echo "=== PHASE 1: Token calibration ==="
$PYTHON "$ROOT_DIR/calibrate_max_tokens.py" \
    --model_path "$MODEL_PATH" \
    --n_samples  10 \
    --ceiling    "$CALIB_CEILING" \
    --gpu_mem    0.90 \
    --max_model_len "$MAX_MODEL_LEN" \
    --output     "$CALIB_JSON"

echo "Calibration complete.  Config: $CALIB_JSON"
cat "$CALIB_JSON"

# ── PHASE 2 — Full inference ──────────────────────────────────────────────────
echo ""
echo "=== PHASE 2: Full inference ==="
mkdir -p "$OUTPUT_DIR"

$PYTHON "$ROOT_DIR/run_inference_benchmarks.py" \
    --model_path    "$MODEL_PATH" \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.90 \
    --batch_size    64 \
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
