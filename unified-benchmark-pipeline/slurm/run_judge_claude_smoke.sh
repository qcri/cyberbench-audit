#!/usr/bin/env bash
#SBATCH --job-name=judge_claude_smoke
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# Sanity test: judge 10 mcq samples from responses_GPT-5.4 with Claude Sonnet 4.6
# (Azure /anthropic/v1/messages). Confirms the new judge path end-to-end on CPU.

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
export PATH="/export/home/aberriche/miniconda3/envs/vllm/bin:$PATH"

mkdir -p "$ROOT_DIR/slurm/logs"

set -a
source "$ROOT_DIR/.env"
set +a

if [ -z "${AZURE_CLAUDE_KEY:-}" ]; then
    echo "ERROR: AZURE_CLAUDE_KEY not set in $ROOT_DIR/.env"
    exit 1
fi

ENDPOINT="https://qcri-claude-hub-cyberxpert.services.ai.azure.com/anthropic/v1/messages"
JUDGE_MODEL="claude-sonnet-4-6-cyberxpert"
RESPONSE_DIR="$ROOT_DIR/outputs/responses_GPT-5.4"
OUT_DIR="$ROOT_DIR/outputs/_smoke_judge_claude/eval_results"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Judge: $JUDGE_MODEL @ $ENDPOINT"
echo "Source responses: $RESPONSE_DIR (mcq, first 10 samples)"
echo "Output: $OUT_DIR"
echo "Start: $(date)"
echo "============================================================"

"$PYTHON" "$ROOT_DIR/run_evaluate_llm_judge.py" \
    --response_dir       "$RESPONSE_DIR" \
    --judge_use_api \
    --judge_api_endpoint "$ENDPOINT" \
    --judge_api_model    "$JUDGE_MODEL" \
    --judge_api_key      "${AZURE_CLAUDE_KEY}" \
    --judge_api_style    "anthropic_messages" \
    --n_workers          4 \
    --tasks              mcq \
    --max_samples        10 \
    --output             "$OUT_DIR"

echo ""
echo "============================================================"
echo "Smoke complete: $OUT_DIR"
echo "End: $(date)"
echo "============================================================"
