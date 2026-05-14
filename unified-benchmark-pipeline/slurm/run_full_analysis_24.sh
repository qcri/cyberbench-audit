#!/usr/bin/env bash
#SBATCH --job-name=analysis_24
#SBATCH --partition=cpu-all
#SBATCH --qos=multi-cpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -uo pipefail

# Comprehensive 24-sub-task analysis re-run.
# Runs (in order): embed (resumable; only cti_taa+sevenllm new) -> correlation
# -> embedding_correlation -> gold_error_voting -> aggregate_verification ->
# all plot generators. Tolerates per-step failures (set -e disabled) so
# downstream plots still run if upstream throws.

ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
PYTHON=/export/home/aberriche/miniconda3/envs/seg_zero/bin/python

cd "$ROOT_DIR"
mkdir -p slurm/logs
export PYTHONPATH="$ROOT_DIR"
export TRANSFORMERS_OFFLINE=0
# Force CPU for sentence-transformers (no CUDA on this partition)
export CUDA_VISIBLE_DEVICES=""

echo "=================================================="
echo "Job ${SLURM_JOB_ID:-local} on $(hostname)  start=$(date)"
echo "=================================================="

step() {
    local name="$1"; shift
    echo ""
    echo "--- step: $name ---"
    "$@"
    local rc=$?
    echo "[$name] exit=$rc"
}

step embed                  "$PYTHON" -u -m analysis.embed --batch-size 32
step correlation            "$PYTHON" -u -m analysis.correlation
step embedding_correlation  "$PYTHON" -u -m analysis.embedding_correlation
step gold_error_voting      "$PYTHON" -u -m analysis.gold_error_voting --aggregate
step aggregate_verification "$PYTHON" -u -m analysis.aggregate_verification
step make_corr_plots        "$PYTHON" -u -m analysis.make_corr_plots
step make_embed_plots       "$PYTHON" -u -m analysis.make_embed_plots
step make_coverage_plots    "$PYTHON" -u -m analysis.make_coverage_plots
step make_verify_plots      "$PYTHON" -u -m analysis.make_verify_plots
step make_plots             "$PYTHON" -u -m analysis.make_plots

echo "=================================================="
echo "Analysis re-run complete  end=$(date)"
echo "=================================================="
