#!/usr/bin/env bash
# Run GPT-5.4 LLM judge evaluation for all models with complete response data.
# No GPU needed — pure API calls. Run on login node or any machine with internet.
#
# Usage:
#   bash run_judge_all_models.sh [--workers N]
#   AZURE_API_KEY and AZURE_JUDGE_ENDPOINT are loaded automatically from .env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python

AZURE_JUDGE_MODEL="gpt-5.4"
N_WORKERS=12

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) N_WORKERS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Load config (API key + endpoint) from .env if not already in environment
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Endpoint can be set in .env as AZURE_JUDGE_ENDPOINT; fallback to the default resource.
AZURE_JUDGE_ENDPOINT="${AZURE_JUDGE_ENDPOINT:-https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/}"

if [ -z "${AZURE_API_KEY:-}" ]; then
    echo "ERROR: AZURE_API_KEY not set and not found in .env"
    exit 1
fi

echo "Judge: $AZURE_JUDGE_MODEL at $AZURE_JUDGE_ENDPOINT"
echo "Workers: $N_WORKERS"
echo "Start: $(date)"
echo ""

# Models with complete (or near-complete) response data
MODELS=(
    "Fanar-2-27B-Instruct"
    "Foundation-Sec-8B-Instruct"
    "GPT-oss-20B"
    "Llama-Primus-Merged"
    "RedSage-Qwen3-8B-DPO"
    "Qwen3.6-35B-A3B"
    "Gemma-4-31B-it"
    "Llama-Primus-Nemotron-70B-Instruct"
    "Llama-3.3-70B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
    RESPONSE_DIR="$SCRIPT_DIR/outputs/responses_${MODEL}"
    JUDGE_DIR="$SCRIPT_DIR/outputs/judge_${MODEL}"

    if [ ! -d "$RESPONSE_DIR" ]; then
        echo "SKIP $MODEL — response dir not found: $RESPONSE_DIR"
        continue
    fi

    N_FILES=$(ls "$RESPONSE_DIR"/*.jsonl 2>/dev/null | wc -l)
    if [ "$N_FILES" -eq 0 ]; then
        echo "SKIP $MODEL — no JSONL files in $RESPONSE_DIR"
        continue
    fi

    echo "======================================================================"
    echo "Judging: $MODEL ($N_FILES task files)"
    echo "======================================================================"
    mkdir -p "$JUDGE_DIR"

    $PYTHON "$SCRIPT_DIR/run_evaluate_llm_judge.py" \
        --response_dir       "$RESPONSE_DIR" \
        --judge_use_api \
        --judge_api_endpoint "$AZURE_JUDGE_ENDPOINT" \
        --judge_api_model    "$AZURE_JUDGE_MODEL" \
        --n_workers          "$N_WORKERS" \
        --output             "$JUDGE_DIR/eval_results"

    echo "Done: $MODEL → $JUDGE_DIR"
    echo ""
done

echo "======================================================================"
echo "All models judged."
echo "End: $(date)"
