#!/usr/bin/env bash
#SBATCH --job-name=infer_claude_smoke
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# 3-sample smoke test for the new anthropic_messages inference path on
# claude-sonnet-4-6-cyberxpert via the Azure pass-through endpoint.

source /export/home/aberriche/miniconda3/etc/profile.d/conda.sh
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks

set -a; [ -f "$ROOT_DIR/.env" ] && source "$ROOT_DIR/.env"; set +a

if [ -z "${AZURE_CLAUDE_KEY:-}" ]; then
    echo "ERROR: AZURE_CLAUDE_KEY not set"
    exit 1
fi

OUT_DIR="$ROOT_DIR/outputs/responses_claude-sonnet-4-6-cyberxpert_smoke"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "=== claude smoke (cti_taa, max_samples=3) ==="
$PYTHON "$ROOT_DIR/run_inference_benchmarks.py" \
    --use_api \
    --api_endpoint  "https://qcri-claude-hub-cyberxpert.services.ai.azure.com/anthropic/v1/messages" \
    --api_model     "claude-sonnet-4-6-cyberxpert" \
    --api_key       "$AZURE_CLAUDE_KEY" \
    --api_style     anthropic_messages \
    --n_api_workers 3 \
    --max_samples   3 \
    --output_dir    "$OUT_DIR" \
    --tasks         cti_taa

echo ""
echo "=== sample[0] ==="
head -1 "$OUT_DIR/cti_taa_responses.jsonl" | $PYTHON -c "
import sys, json
d = json.loads(sys.stdin.read())
print('task:', d.get('task'))
print('idx:', d.get('index'))
print('response[:300]:', d.get('model_response','')[:300])
"
