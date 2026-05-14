#!/usr/bin/env bash
#SBATCH --job-name=judge_claude_one
#SBATCH --partition=gpu-all
#SBATCH --qos=20gpus
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# Per-model Claude Sonnet 4.6 judge run on the gpu-all partition (CPU-only via gres=gpu:0).
# Designed to be fanned out — one sbatch per evaluated model — so multiple models judge in
# parallel against the shared Anthropic deployment.
#
# Usage:
#   sbatch --export=ALL,MODEL_NAME=Llama-3.3-70B-Instruct slurm/run_judge_claude_one.sh
#
# Skip-if-exists per task: safe to resubmit; previously-judged tasks are skipped instantly.

if [ -z "${MODEL_NAME:-}" ]; then
    echo "ERROR: MODEL_NAME not set. Use: sbatch --export=ALL,MODEL_NAME=<name> $0" >&2
    exit 1
fi

ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
export PATH="/export/home/aberriche/miniconda3/envs/vllm/bin:$PATH"

mkdir -p "$ROOT_DIR/slurm/logs"

set -a
source "$ROOT_DIR/.env"
set +a

if [ -z "${AZURE_CLAUDE_KEY:-}" ]; then
    echo "ERROR: AZURE_CLAUDE_KEY not set in $ROOT_DIR/.env" >&2
    exit 1
fi

ENDPOINT="https://qcri-claude-hub-cyberxpert.services.ai.azure.com/anthropic/v1/messages"
JUDGE_MODEL="claude-sonnet-4-6-cyberxpert"
JUDGE_ALIAS="claude-sonnet-4-6"
N_WORKERS="${N_WORKERS:-8}"

ALL_TASKS="mcq rcm vsp ate athena_rcm athena_vsp athena_ate ckt rms taa cybermetric mmlu_cs secbench seceval redsage_cli redsage_frameworks redsage_generals redsage_kali redsage_skills secure_cwet secure_kcv secure_maet"

RESPONSE_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"
JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL_NAME}__by_${JUDGE_ALIAS}/eval_results"

if [ ! -d "$RESPONSE_DIR" ]; then
    echo "ERROR: response dir not found: $RESPONSE_DIR" >&2
    exit 1
fi

mkdir -p "$JUDGE_DIR"

# Determine which tasks still need judging
TASKS=""
for t in $ALL_TASKS; do
    if [ ! -f "$JUDGE_DIR/${t}_detailed.jsonl" ]; then
        TASKS="$TASKS $t"
    fi
done
TASKS=$(echo "$TASKS" | xargs)

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Model: $MODEL_NAME"
echo "Judge: $JUDGE_MODEL @ $ENDPOINT"
echo "Workers: $N_WORKERS"
echo "Output: $JUDGE_DIR"
echo "Pending tasks: $TASKS"
echo "Start: $(date)"
echo "============================================================"

if [ -z "$TASKS" ]; then
    echo "Nothing to do — all 22 tasks already judged."
    exit 0
fi

"$PYTHON" "$ROOT_DIR/run_evaluate_llm_judge.py" \
    --response_dir       "$RESPONSE_DIR" \
    --judge_use_api \
    --judge_api_endpoint "$ENDPOINT" \
    --judge_api_model    "$JUDGE_MODEL" \
    --judge_api_key      "${AZURE_CLAUDE_KEY}" \
    --judge_api_style    "anthropic_messages" \
    --n_workers          "$N_WORKERS" \
    --tasks              $TASKS \
    --output             "$JUDGE_DIR"

echo "============================================================"
echo "[$MODEL_NAME] done at $(date)"
echo "============================================================"
