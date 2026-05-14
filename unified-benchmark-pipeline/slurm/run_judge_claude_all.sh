#!/usr/bin/env bash
#SBATCH --job-name=judge_claude_all
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -euo pipefail

# Full Claude Sonnet 4.6 judge sweep over all 10 evaluated models × 22 tasks.
# Wraps run_judge_unified_all_claude.sh — that script handles per-task skip-if-exists,
# so this slurm job is safely re-submittable to resume after interruption.

ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
export PATH="/export/home/aberriche/miniconda3/envs/vllm/bin:$PATH"

mkdir -p "$ROOT_DIR/slurm/logs"

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Driver: $ROOT_DIR/run_judge_unified_all_claude.sh"
echo "Start: $(date)"
echo "============================================================"

bash "$ROOT_DIR/run_judge_unified_all_claude.sh"

echo "============================================================"
echo "All-Claude sweep finished at $(date)"
echo "============================================================"
