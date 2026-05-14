#!/usr/bin/env bash
#SBATCH --job-name=verify_search_t50
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=23:50:00
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/%x_%j.err
set -uo pipefail   # not -e — verify.py prints + continues on per-sample errors

# Resumable verification of the τ≥0.5 flagged-bank using the search-grounded
# GPT-5.4 agent only. The previous τ≥0.75 verdicts are already cached under
# verification/verdicts/search/, so they are skipped automatically.
#
# Submit once. If wall-time elapses before the bank is exhausted, just submit
# again — the cache picks up where we left off.

source /export/home/aberriche/miniconda3/etc/profile.d/conda.sh
conda activate vllm

PYTHON=/export/home/aberriche/miniconda3/envs/vllm/bin/python
ROOT_DIR=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
mkdir -p "$ROOT_DIR/slurm/logs"

set -a; [ -f "$ROOT_DIR/.env" ] && source "$ROOT_DIR/.env"; set +a

if [ -z "${AZURE_API_KEY:-}" ]; then
    echo "ERROR: AZURE_API_KEY not set"
    exit 1
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

echo "============================================================"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-local}"
echo "Cached search verdicts before this run:"
ls "$ROOT_DIR/analysis/reports/verification/verdicts/search/" 2>/dev/null | wc -l
echo "Start: $(date)"
echo "============================================================"

# 23h45m budget inside the script so the job exits cleanly before the slurm
# wall-time kicks in; if more remain, just resubmit.
$PYTHON -m analysis.verify \
    --agent search \
    --max-threshold 0.5 \
    --time-budget-s 85500

echo ""
echo "============================================================"
echo "Cached search verdicts after this run:"
ls "$ROOT_DIR/analysis/reports/verification/verdicts/search/" 2>/dev/null | wc -l
echo "End: $(date)"
echo "============================================================"
