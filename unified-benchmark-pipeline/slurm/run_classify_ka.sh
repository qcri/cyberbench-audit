#!/usr/bin/env bash
#SBATCH --job-name=ka_classify
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/ka_%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/ka_%x_%j.err

# Usage:
#   API (cpu) — uses Azure GPT-5.4 (our top-1 model):
#     sbatch --partition=cpu-all --qos=multi-cpus --time=01:00:00 --cpus-per-task=2 --mem=4G \
#            --export=ALL,BACKEND=api,MODEL_LABEL=GPT-5.4 slurm/run_classify_ka.sh
#
#   vLLM (gpu) — top-3 / top-4 / substitute models:
#     sbatch --partition=gpu-all --qos=20gpus --gres=gpu:1 --time=01:00:00 --cpus-per-task=4 --mem=64G \
#            --export=ALL,BACKEND=vllm,MODEL_LABEL=Qwen3.6-35B-A3B,MODEL_PATH=Qwen/Qwen3.6-35B-A3B \
#            slurm/run_classify_ka.sh

set -euo pipefail

ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python

cd "$ROOT_DIR"
mkdir -p slurm/logs

set -a
source "$ROOT_DIR/.env"
set +a

export HF_HOME=/export/cyb-ai-research-data/aberriche/hf_cache
export PATH="/export/home/aberriche/miniconda3/envs/vllm/bin:$PATH"

BACKEND="${BACKEND:?must set BACKEND=api or BACKEND=vllm}"
MODEL_LABEL="${MODEL_LABEL:?must set MODEL_LABEL}"
TIME_BUDGET="${TIME_BUDGET:-3000}"

echo "=== ka_classify ==="
echo "host=$(hostname)  start=$(date)"
echo "backend=$BACKEND  model=$MODEL_LABEL"

if [ "$BACKEND" = "vllm" ]; then
    if [ -z "${MODEL_PATH:-}" ]; then
        echo "ERROR: MODEL_PATH must be set for backend=vllm" >&2
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader || true
    PYTHONPATH=. "$PYTHON" -u -m analysis.classify_ka \
        --backend vllm --model-label "$MODEL_LABEL" --model-path "$MODEL_PATH"
else
    PYTHONPATH=. "$PYTHON" -u -m analysis.classify_ka \
        --backend api --model-label "$MODEL_LABEL" --time-budget-s "$TIME_BUDGET"
fi

echo "end=$(date)"
