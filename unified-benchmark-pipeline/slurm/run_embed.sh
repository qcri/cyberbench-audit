#!/usr/bin/env bash
#SBATCH --job-name=embed_bench
#SBATCH --partition=gpu-all
#SBATCH --qos=20gpus
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/embed_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/embed_%j.err

set -euo pipefail

ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
PYTHON=/export/home/aberriche/miniconda3/envs/seg_zero/bin/python

cd "$ROOT_DIR"
mkdir -p slurm/logs

MODEL="${MODEL:-BAAI/bge-base-en-v1.5}"
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "=== embed run ==="
echo "model=$MODEL  batch=$BATCH_SIZE  host=$(hostname)  start=$(date)"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "(no gpu)"

PYTHONPATH=. "$PYTHON" -u -m analysis.embed --model "$MODEL" --batch-size "$BATCH_SIZE"

echo "end=$(date)"
