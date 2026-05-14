#!/usr/bin/env bash
# Submit local cti_taa + sevenllm judging for every model that has both
# response files in outputs/responses_<MODEL>/.
#
# Usage: bash launch_judge_new_tasks_all.sh

set -euo pipefail

ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
SCRIPT="$ROOT_DIR/slurm/run_judge_local_new_tasks.sh"

cd "$ROOT_DIR/outputs"
MODELS=()
for d in responses_*/; do
    name="${d%/}"
    name="${name#responses_}"
    # skip the v2 symlink dirs and the smoke test dir
    case "$name" in
        *_v2|*_smoke) continue ;;
    esac
    if [ -f "responses_${name}/cti_taa_responses.jsonl" ] || [ -f "responses_${name}/sevenllm_responses.jsonl" ]; then
        MODELS+=("$name")
    fi
done

echo "Launching cti_taa+sevenllm judge for ${#MODELS[@]} models..."
echo

for NAME in "${MODELS[@]}"; do
    JOBID=$(sbatch --parsable \
        --job-name="judge_${NAME}_new_tasks" \
        --export="ALL,MODEL_NAME=${NAME}" \
        "$SCRIPT")
    echo "  [$JOBID] $NAME"
done

echo
echo "Track with: squeue -u \$USER | grep new_tasks"
