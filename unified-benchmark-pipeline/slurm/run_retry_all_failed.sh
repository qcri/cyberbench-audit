#!/usr/bin/env bash
#SBATCH --job-name=retry_all_failed
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -uo pipefail   # no -e: keep going across per-model failures

# Phase A: inference retries (only GPT-5.4 has ERRORs — 132 across 5 tasks)
# Phase B: judge retries for every model with skipped:true rows in any
#          detailed.jsonl. Re-judges only those rows in place and rewrites
#          summary.json from refreshed verdicts.

source /export/home/aberriche/miniconda3/etc/profile.d/conda.sh
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
mkdir -p "$ROOT_DIR/slurm/logs"

if [ -z "${AZURE_API_KEY:-}" ] && [ -f "$ROOT_DIR/.env" ]; then
    set -a; source "$ROOT_DIR/.env"; set +a
fi

API_ENDPOINT="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/"
API_MODEL="gpt-5.4"

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-local}"
echo "Start: $(date)"
echo "============================================================"

# ── Phase A: inference retries (GPT-5.4 only — others use vLLM, can't be
#    retried in-place from cpu-all without spinning up GPUs).
echo ""
echo "=== Phase A: inference retries (GPT-5.4) ==="
$PYTHON "$ROOT_DIR/retry_failed_samples.py" \
    --response_dir "$ROOT_DIR/outputs/responses_GPT-5.4" \
    --tasks cybermetric secure_kcv taa redsage_generals redsage_kali \
    --api_endpoint "$API_ENDPOINT" \
    --api_model "$API_MODEL" \
    --api_key "$AZURE_API_KEY" \
    --workers 8

# ── Phase B: judge retries for every model.
echo ""
echo "=== Phase B: judge retries (all models) ==="

# All models with a default judge dir; we skip _v1 and _v2 by listing only
# those without a suffix.
MODELS=(
    "Fanar-2-27B-Instruct"
    "Foundation-Sec-8B-Instruct"
    "Gemma-4-31B-it"
    "GPT-5.4"
    "GPT-oss-20B"
    "Llama-3.3-70B-Instruct"
    "Llama-Primus-Merged"
    "Llama-Primus-Nemotron-70B-Instruct"
    "Qwen3.6-35B-A3B"
    "RedSage-Qwen3-8B-DPO"
)

for MODEL in "${MODELS[@]}"; do
    JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL}/eval_results"
    RESP_DIR="$ROOT_DIR/outputs/responses_${MODEL}"
    if [ ! -d "$JUDGE_DIR" ] || [ ! -d "$RESP_DIR" ]; then
        echo "[$MODEL] missing dir — skipping"
        continue
    fi
    echo ""
    echo "----- $MODEL -----"
    $PYTHON "$ROOT_DIR/retry_failed_judges.py" \
        --judge_dir          "$JUDGE_DIR" \
        --response_dir       "$RESP_DIR" \
        --judge_api_endpoint "$API_ENDPOINT" \
        --judge_api_model    "$API_MODEL" \
        --judge_api_key      "$AZURE_API_KEY" \
        --workers            16
done

# ── Phase C: optional inference re-judging for GPT-5.4 tasks whose responses
#    just got fixed in Phase A. Force a fresh judge by deleting the detailed
#    files so we don't leave stale (skipped or pre-fix) verdicts.
echo ""
echo "=== Phase C: re-judge GPT-5.4 tasks whose inference was fixed ==="
GPT5_JUDGE="$ROOT_DIR/outputs/judge_GPT-5.4/eval_results"
GPT5_RESP="$ROOT_DIR/outputs/responses_GPT-5.4"
GPT5_TASKS="cybermetric secure_kcv taa redsage_generals redsage_kali"

for t in $GPT5_TASKS; do
    rm -f "$GPT5_JUDGE/${t}_detailed.jsonl"
done

$PYTHON "$ROOT_DIR/run_evaluate_llm_judge.py" \
    --response_dir       "$GPT5_RESP" \
    --judge_use_api \
    --judge_api_endpoint "$API_ENDPOINT" \
    --judge_api_model    "$API_MODEL" \
    --judge_api_key      "$AZURE_API_KEY" \
    --n_workers          8 \
    --tasks              $GPT5_TASKS \
    --output             "$GPT5_JUDGE"

echo ""
echo "============================================================"
echo "End: $(date)"
echo "============================================================"
