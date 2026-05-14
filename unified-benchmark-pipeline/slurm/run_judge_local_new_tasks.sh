#!/usr/bin/env bash
#SBATCH --job-name=judge_local_new_tasks
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# Judge cti_taa + sevenllm responses produced by our local inference, written
# to outputs/judge_<MODEL>/eval_results/ (the default judge dir, NOT _v2).
# Skip-if-exists keeps the 22 existing task verdicts intact.
#
# Usage: sbatch --export=ALL,MODEL_NAME=<name> run_judge_local_new_tasks.sh

if [ -z "${MODEL_NAME:-}" ]; then
    echo "ERROR: MODEL_NAME is not set. Use: sbatch --export=ALL,MODEL_NAME=<name> $0"
    exit 1
fi

source /export/home/aberriche/miniconda3/etc/profile.d/conda.sh
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks

mkdir -p "$ROOT_DIR/slurm/logs"

if [ -f "$ROOT_DIR/.env" ]; then
    set -a; source "$ROOT_DIR/.env"; set +a
fi

if [ -z "${AZURE_API_KEY:-}" ]; then
    echo "ERROR: AZURE_API_KEY not set and not found in .env"
    exit 1
fi

ENDPOINT="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/"
JUDGE_MODEL="gpt-5.4"
N_WORKERS=8

RESPONSE_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}"
JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL_NAME}/eval_results"

if [ ! -d "$RESPONSE_DIR" ]; then
    echo "ERROR: response dir missing: $RESPONSE_DIR"
    exit 1
fi

mkdir -p "$JUDGE_DIR"

# Only the two new tasks. skip-if-exists prevents accidental rejudging.
TASKS="cti_taa sevenllm"
TODO=""
for t in $TASKS; do
    if [ ! -f "$JUDGE_DIR/${t}_detailed.jsonl" ]; then
        TODO="$TODO $t"
    fi
done
TODO=$(echo "$TODO" | xargs)

if [ -z "$TODO" ]; then
    echo "[$MODEL_NAME] cti_taa and sevenllm already judged — nothing to do."
    exit 0
fi

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-local}"
echo "Model: $MODEL_NAME (default judge dir, new tasks only)"
echo "Response dir: $RESPONSE_DIR"
echo "Judge dir:    $JUDGE_DIR"
echo "Tasks: $TODO"
echo "Workers: $N_WORKERS"
echo "Start: $(date)"
echo "============================================================"

$PYTHON "$ROOT_DIR/run_evaluate_llm_judge.py" \
    --response_dir       "$RESPONSE_DIR" \
    --judge_use_api \
    --judge_api_endpoint "$ENDPOINT" \
    --judge_api_model    "$JUDGE_MODEL" \
    --judge_api_key      "${AZURE_API_KEY}" \
    --n_workers          "$N_WORKERS" \
    --tasks              $TODO \
    --output             "$JUDGE_DIR"

echo "============================================================"
echo "[$MODEL_NAME] new-task judging complete: $JUDGE_DIR"
echo "End: $(date)"
echo "============================================================"
