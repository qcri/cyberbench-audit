#!/usr/bin/env bash
#SBATCH --job-name=verify_gpt54
#SBATCH --partition=cpu-all
#SBATCH --qos=multi-cpus
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/verify_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/verify_%j.err

# Usage:
#   sbatch --export=ALL,AGENT=direct,MAX_THRESHOLD=1.0 run_verify.sh
#   sbatch --export=ALL,AGENT=search,MAX_THRESHOLD=1.0 run_verify.sh
#   sbatch --export=ALL,AGENT=both,MAX_THRESHOLD=0.75  run_verify.sh

set -euo pipefail

ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python

cd "$ROOT_DIR"
mkdir -p slurm/logs

set -a
source "$ROOT_DIR/.env"
set +a

AGENT="${AGENT:-both}"
MAX_THR="${MAX_THRESHOLD:-1.0}"
TIME_BUDGET="${TIME_BUDGET:-13000}"   # ~3.6h; slurm wallclock is 4h

echo "=== verify run ==="
echo "agent=$AGENT  max_threshold=$MAX_THR  time_budget=${TIME_BUDGET}s"
echo "host=$(hostname)  start=$(date)"

PYTHONPATH=. "$PYTHON" -u -m analysis.verify \
    --agent "$AGENT" \
    --max-threshold "$MAX_THR" \
    --time-budget-s "$TIME_BUDGET"

echo "end=$(date)"
