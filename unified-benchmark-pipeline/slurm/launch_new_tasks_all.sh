#!/usr/bin/env bash
# Submit cti_taa + sevenllm inference for every locally-tracked model.
# Each model gets its own slurm job; --skip_completed protects existing tasks.
# Skips API-only models (GPT-5.4) — those need a different inference path.
#
# Usage: bash launch_new_tasks_all.sh

set -euo pipefail

ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
MODELS_DIR=/export/cyb-ai-research-data/aberriche/models
SCRIPT="$ROOT_DIR/slurm/run_inference_new_tasks.sh"

# (MODEL_NAME, primary path, HF fallback)
# Lines starting with # are skipped.
MODELS=(
    "Fanar-2-27B-Instruct|$MODELS_DIR/Fanar-2-27B-Instruct|QCRI/Fanar-2-27B-Instruct"
    "Foundation-Sec-8B-Instruct|$MODELS_DIR/Foundation-Sec-8B-Instruct|fdtn-ai/Foundation-Sec-8B-Instruct"
    "Gemma-4-31B-it|/export/cyb-ai-research-data/cshalby/models/gemma-4-31B-it|google/gemma-4-31B"
    "GPT-oss-20B|$MODELS_DIR/gpt-oss-20b|openai/gpt-oss-20b"
    "Llama-3.3-70B-Instruct|$MODELS_DIR/Llama-3.3-70B-Instruct|meta-llama/Llama-3.3-70B-Instruct"
    "Llama-Primus-Merged|$MODELS_DIR/Llama-Primus-Merged|trendmicro-ailab/Llama-Primus-Merged"
    "Llama-Primus-Nemotron-70B-Instruct|/export/cyb-ai-research-data/cshalby/models/Llama-Primus-Nemotron-70B-Instruct|"
    "Qwen3.6-35B-A3B|$MODELS_DIR/Qwen3.6-35B-A3B|Qwen/Qwen3.6-35B-A3B"
    "RedSage-Qwen3-8B-DPO|$MODELS_DIR/RedSage-Qwen3-8B-DPO|RISys-Lab/RedSage-Qwen3-8B-DPO"
    # GPT-5.4 — API model, vLLM path doesn't apply. Run separately if needed.
)

echo "Launching cti_taa + sevenllm inference for ${#MODELS[@]} models..."
echo

for entry in "${MODELS[@]}"; do
    IFS='|' read -r NAME PRIMARY FALLBACK <<< "$entry"

    # Pick existing path: primary if present, else fallback HF id
    if [ -d "$PRIMARY" ]; then
        MPATH="$PRIMARY"
    elif [ -n "$FALLBACK" ]; then
        MPATH="$FALLBACK"
    else
        MPATH="$PRIMARY"
    fi

    JOBID=$(sbatch --parsable \
        --job-name="infer_${NAME}_new_tasks" \
        --export="ALL,MODEL_NAME=${NAME},MODEL_PATH=${MPATH}" \
        "$SCRIPT")
    echo "  [$JOBID] $NAME -> $MPATH"
done

echo
echo "All jobs submitted. Track with: squeue -u \$USER | grep new_tasks"
