#!/usr/bin/env bash
#SBATCH --job-name=infer_gpt5_new_tasks
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# Run GPT-5.4 inference for cti_taa + sevenllm only (Azure API, cpu-only),
# appending to outputs/responses_GPT-5.4/. --skip_completed protects existing
# results. Then judge those new responses into outputs/judge_GPT-5.4/eval_results/.

source /export/home/aberriche/miniconda3/etc/profile.d/conda.sh
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
mkdir -p "$ROOT_DIR/slurm/logs"

if [ -z "${AZURE_API_KEY:-}" ] && [ -f "$ROOT_DIR/.env" ]; then
    set -a; source "$ROOT_DIR/.env"; set +a
fi
if [ -z "${AZURE_API_KEY:-}" ]; then
    echo "ERROR: AZURE_API_KEY not set"
    exit 1
fi

MODEL_NAME="GPT-5.4"
API_ENDPOINT="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/"
API_MODEL="gpt-5.4"

OUTPUT_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"
JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL_NAME}/eval_results"

mkdir -p "$OUTPUT_DIR" "$JUDGE_DIR"

TASKS="cti_taa sevenllm"

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-local}"
echo "Model: $MODEL_NAME (Azure API, new tasks only)"
echo "Tasks: $TASKS"
echo "Output: $OUTPUT_DIR"
echo "Judge:  $JUDGE_DIR"
echo "Start:  $(date)"
echo "============================================================"

# ── Phase 1 — Inference via Azure API ────────────────────────────────────────
echo ""
echo "=== Phase 1: API inference ==="
$PYTHON "$ROOT_DIR/run_inference_benchmarks.py" \
    --use_api \
    --api_endpoint  "$API_ENDPOINT" \
    --api_model     "$API_MODEL" \
    --api_key       "$AZURE_API_KEY" \
    --n_api_workers 8 \
    --output_dir    "$OUTPUT_DIR" \
    --tasks         $TASKS \
    --skip_completed

# ── Phase 2 — Judging ─────────────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Judging the new tasks ==="

# Skip-if-exists at the judge level too: only judge the tasks not already done
TODO=""
for t in $TASKS; do
    if [ ! -f "$JUDGE_DIR/${t}_detailed.jsonl" ]; then
        TODO="$TODO $t"
    fi
done
TODO=$(echo "$TODO" | xargs)

if [ -z "$TODO" ]; then
    echo "[GPT-5.4] cti_taa and sevenllm already judged — nothing to do."
else
    $PYTHON "$ROOT_DIR/run_evaluate_llm_judge.py" \
        --response_dir       "$OUTPUT_DIR" \
        --judge_use_api \
        --judge_api_endpoint "$API_ENDPOINT" \
        --judge_api_model    "$API_MODEL" \
        --judge_api_key      "${AZURE_API_KEY}" \
        --n_workers          8 \
        --tasks              $TODO \
        --output             "$JUDGE_DIR"
fi

echo ""
echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
