#!/usr/bin/env bash
#SBATCH --job-name=judge_v2
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# Usage: sbatch --export=ALL,MODEL_NAME=<base-model-name> run_judge_v2.sh
# Judges new inference outputs at
#   /export/cyb-ai-research-data/benchmarking_sec/raw_results_original/<MODEL_NAME>/inference_responses/
# Outputs to outputs/judge_<MODEL_NAME>_v2/eval_results/.
# Task list matches v1 (skips cybermetric_paper, mmlu-cs-logprobs, sevenllm).

if [ -z "${MODEL_NAME:-}" ]; then
    echo "ERROR: MODEL_NAME is not set. Use: sbatch --export=ALL,MODEL_NAME=<name> $0"
    exit 1
fi

source /export/home/aberriche/miniconda3/etc/profile.d/conda.sh
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
RAW_DIR=/export/cyb-ai-research-data/benchmarking_sec/raw_results_original

mkdir -p "$ROOT_DIR/slurm/logs"

if [ -f "$ROOT_DIR/.env" ]; then
    set -a; source "$ROOT_DIR/.env"; set +a
fi

if [ -z "${AZURE_API_KEY:-}" ]; then
    echo "ERROR: AZURE_API_KEY not set and not found in .env"
    exit 1
fi

AZURE_JUDGE_ENDPOINT="https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/"
AZURE_JUDGE_MODEL="gpt-5.4"
N_WORKERS=8

RESPONSE_SRC="$RAW_DIR/${MODEL_NAME}/inference_responses"
RESPONSE_DIR="$ROOT_DIR/outputs/responses_${MODEL_NAME}_v2"
JUDGE_DIR="$ROOT_DIR/outputs/judge_${MODEL_NAME}_v2/eval_results"

if [ ! -d "$RESPONSE_SRC" ]; then
    echo "ERROR: Source response dir not found: $RESPONSE_SRC"
    exit 1
fi

# Symlink new inference outputs into outputs/ tree (idempotent)
if [ ! -e "$RESPONSE_DIR" ]; then
    ln -s "$RESPONSE_SRC" "$RESPONSE_DIR"
fi

mkdir -p "$JUDGE_DIR"

# Full 26-task list (v1 22 + 4 new: cti_taa, cybermetric_paper, mmlu_cs_logprobs, sevenllm).
# Tasks not present as response files are auto-skipped by the judge script.
ALL_TASKS="mcq rcm vsp ate athena_rcm athena_vsp athena_ate ckt rms taa cti_taa cybermetric cybermetric_paper mmlu_cs mmlu_cs_logprobs secbench seceval sevenllm redsage_cli redsage_frameworks redsage_generals redsage_kali redsage_skills secure_cwet secure_kcv secure_maet"

# Skip-if-exists: only judge tasks not already done
TASKS=""
for t in $ALL_TASKS; do
    if [ ! -f "$JUDGE_DIR/${t}_detailed.jsonl" ]; then
        TASKS="$TASKS $t"
    fi
done
TASKS=$(echo "$TASKS" | xargs)

if [ -z "$TASKS" ]; then
    echo "[$MODEL_NAME] all tasks already judged — nothing to do."
    exit 0
fi

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-local}"
echo "Model: $MODEL_NAME (v2)"
echo "Source: $RESPONSE_SRC"
echo "Symlink: $RESPONSE_DIR"
echo "Judge output: $JUDGE_DIR"
echo "Tasks ($(echo $TASKS | wc -w)): $TASKS"
echo "Workers: $N_WORKERS"
echo "Start: $(date)"
echo "============================================================"

$PYTHON "$ROOT_DIR/run_evaluate_llm_judge.py" \
    --response_dir       "$RESPONSE_DIR" \
    --judge_use_api \
    --judge_api_endpoint "$AZURE_JUDGE_ENDPOINT" \
    --judge_api_model    "$AZURE_JUDGE_MODEL" \
    --judge_api_key      "${AZURE_API_KEY}" \
    --n_workers          "$N_WORKERS" \
    --tasks              $TASKS \
    --output             "$JUDGE_DIR"

echo "============================================================"
echo "[$MODEL_NAME] judge v2 complete: $JUDGE_DIR"
echo "End: $(date)"
echo "============================================================"
