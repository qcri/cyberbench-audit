#!/usr/bin/env bash
#SBATCH --job-name=analysis
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# Run a chained analysis pipeline on cpu-all (heavy login-node restrictions
# kill big-IO scripts). Targets:
#   1. analysis.judge_agreement   — 3-judge agreement report
#   2. analysis.results_table     — master accuracy table (majority vote)
#
# Override TARGETS env var to run a single step:
#   sbatch --export=ALL,TARGETS=judge_agreement run_analysis.sh

source /export/home/aberriche/miniconda3/etc/profile.d/conda.sh
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

mkdir -p "$ROOT_DIR/slurm/logs"

TARGETS="${TARGETS:-judge_agreement results_table}"

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-local}"
echo "Targets: $TARGETS"
echo "Start: $(date)"
echo "============================================================"

for tgt in $TARGETS; do
    echo ""
    echo "=== analysis.$tgt ==="
    "$PYTHON" -m "analysis.$tgt"
    echo "[$tgt] exit=$?"
done

echo ""
echo "============================================================"
echo "Analysis complete."
echo "End: $(date)"
echo "============================================================"
