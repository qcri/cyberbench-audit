#!/usr/bin/env bash
#SBATCH --job-name=retry_gpt5_failed
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# Retry the cti_taa+sevenllm GPT-5.4 samples that previously got ERROR
# (content_filter), then re-judge only those task files.

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
RESPONSE_DIR="$ROOT_DIR/outputs/responses_GPT-5.4"
JUDGE_DIR="$ROOT_DIR/outputs/judge_GPT-5.4/eval_results"

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-local}"
echo "Retrying previously-blocked GPT-5.4 cti_taa + sevenllm samples"
echo "Start: $(date)"
echo "============================================================"

# 1) Retry the ERROR samples in-place
$PYTHON "$ROOT_DIR/retry_failed_samples.py" \
    --response_dir "$RESPONSE_DIR" \
    --tasks cti_taa sevenllm \
    --api_endpoint "$API_ENDPOINT" \
    --api_model "$API_MODEL" \
    --api_key "$AZURE_API_KEY" \
    --workers 4

# 2) Re-judge — delete existing detailed.jsonl so the judge re-runs both tasks
#    fresh (sample indices won't change but the new responses must be evaluated).
echo ""
echo "=== Re-running judge for cti_taa + sevenllm ==="
rm -f "$JUDGE_DIR/cti_taa_detailed.jsonl" "$JUDGE_DIR/sevenllm_detailed.jsonl"

$PYTHON "$ROOT_DIR/run_evaluate_llm_judge.py" \
    --response_dir       "$RESPONSE_DIR" \
    --judge_use_api \
    --judge_api_endpoint "$API_ENDPOINT" \
    --judge_api_model    "$API_MODEL" \
    --judge_api_key      "$AZURE_API_KEY" \
    --n_workers          8 \
    --tasks              cti_taa sevenllm \
    --output             "$JUDGE_DIR"

echo ""
echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
