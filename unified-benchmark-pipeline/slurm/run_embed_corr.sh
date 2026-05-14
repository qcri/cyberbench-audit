#!/usr/bin/env bash
#SBATCH --job-name=embed_corr
#SBATCH --partition=cpu-all
#SBATCH --qos=multi-cpus
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/embed_corr_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/embed_corr_%j.err

set -euo pipefail

ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
PYTHON=/export/home/aberriche/miniconda3/envs/seg_zero/bin/python

cd "$ROOT_DIR"
mkdir -p slurm/logs

echo "=== embedding_correlation ==="
echo "host=$(hostname)  start=$(date)"
PYTHONPATH=. "$PYTHON" -u -m analysis.embedding_correlation
echo "=== make_embed_plots ==="
PYTHONPATH=. "$PYTHON" -u -m analysis.make_embed_plots
echo "end=$(date)"
