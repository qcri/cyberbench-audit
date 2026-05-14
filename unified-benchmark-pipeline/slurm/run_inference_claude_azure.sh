#!/usr/bin/env bash
#SBATCH --job-name=infer_claude_azure
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# Run claude-sonnet-4-6-cyberxpert inference for the full local-pipeline task
# set, via the Azure-pass-through Anthropic messages API. CPU-only (no GPU
# needed for API calls). Then judge the new responses with GPT-5.4.
#
# Submit:
#   sbatch slurm/run_inference_claude_azure.sh
#
# Override TASKS via env to run a subset:
#   sbatch --export=ALL,TASKS="cti_taa sevenllm" slurm/run_inference_claude_azure.sh

source /export/home/aberriche/miniconda3/etc/profile.d/conda.sh
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
mkdir -p "$ROOT_DIR/slurm/logs"

set -a; [ -f "$ROOT_DIR/.env" ] && source "$ROOT_DIR/.env"; set +a

if [ -z "${AZURE_CLAUDE_KEY:-}" ]; then
    echo "ERROR: AZURE_CLAUDE_KEY not set in $ROOT_DIR/.env" >&2
    exit 1
fi
if [ -z "${AZURE_API_KEY:-}" ]; then
    echo "ERROR: AZURE_API_KEY not set (needed for Phase 3 judge)" >&2
    exit 1
fi

# ── Inference config (Azure-pass-through Anthropic messages) ────────────────
MODEL_NAME="claude-sonnet-4-6-cyberxpert"
INFER_ENDPOINT="https://qcri-claude-hub-cyberxpert.services.ai.azure.com/anthropic/v1/messages"
INFER_MODEL="claude-sonnet-4-6-cyberxpert"

# ── Judge config (GPT-5.4 Azure OpenAI) ──────────────────────────────────────
JUDGE_ENDPOINT="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/"
JUDGE_MODEL="gpt-5.4"

OUTPUT_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"
JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL_NAME}/eval_results"
mkdir -p "$OUTPUT_DIR" "$JUDGE_DIR"

# Default task set = the 24-task local pipeline (matches loaders.TASK_ORDER)
TASKS="${TASKS:-mcq rcm vsp ate cti_taa ckt rms taa athena_ate athena_rcm athena_vsp \
                secure_maet secure_cwet secure_kcv \
                seceval cybermetric mmlu-cs secbench \
                redsage_frameworks redsage_generals redsage_skills \
                redsage_cli redsage_kali sevenllm}"

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-local}"
echo "Model: $MODEL_NAME (Anthropic messages via Azure)"
echo "Inference endpoint: $INFER_ENDPOINT"
echo "Judge endpoint:     $JUDGE_ENDPOINT"
echo "Tasks: $TASKS"
echo "Start: $(date)"
echo "============================================================"

# ── Phase 1 — Inference ──────────────────────────────────────────────────────
echo ""
echo "=== Phase 1: Anthropic-API inference ==="
$PYTHON "$ROOT_DIR/run_inference_benchmarks.py" \
    --use_api \
    --api_endpoint  "$INFER_ENDPOINT" \
    --api_model     "$INFER_MODEL" \
    --api_key       "$AZURE_CLAUDE_KEY" \
    --api_style     anthropic_messages \
    --n_api_workers 8 \
    --output_dir    "$OUTPUT_DIR" \
    --tasks         $TASKS \
    --skip_completed

# ── Phase 2 — Judging via GPT-5.4 ────────────────────────────────────────────
echo ""
echo "=== Phase 2: GPT-5.4 judging ==="

# skip-if-exists at task level
TASKS_LOWER=$(echo "$TASKS" | tr ' ' '\n' | sed 's/-/_/g' | tr '\n' ' ')
TODO=""
for t in $TASKS_LOWER; do
    if [ ! -f "$JUDGE_DIR/${t}_detailed.jsonl" ]; then
        TODO="$TODO $t"
    fi
done
TODO=$(echo "$TODO" | xargs)

if [ -z "$TODO" ]; then
    echo "[$MODEL_NAME] all tasks already judged — nothing to do."
else
    $PYTHON "$ROOT_DIR/run_evaluate_llm_judge.py" \
        --response_dir       "$OUTPUT_DIR" \
        --judge_use_api \
        --judge_api_endpoint "$JUDGE_ENDPOINT" \
        --judge_api_model    "$JUDGE_MODEL" \
        --judge_api_key      "$AZURE_API_KEY" \
        --n_workers          8 \
        --tasks              $TODO \
        --output             "$JUDGE_DIR"
fi

echo ""
echo "============================================================"
echo "End: $(date)"
echo "============================================================"
