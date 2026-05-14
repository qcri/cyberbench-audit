#!/usr/bin/env bash
#SBATCH --job-name=reparse_apply
#SBATCH --output=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/reparse_apply_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/logs/reparse_apply_%j.err
#SBATCH --partition=gpu-all
#SBATCH --qos=20gpus
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:0
cd /export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
/export/home/aberriche/miniconda3/envs/vllm/bin/python /export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/slurm/reparse_qwen.py --apply
echo "---aggregating---"
PYTHONPATH=. /export/home/aberriche/miniconda3/envs/vllm/bin/python -u -m analysis.aggregate_ka
echo "---plotting---"
PYTHONPATH=. /export/home/aberriche/miniconda3/envs/seg_zero/bin/python -u -m analysis.make_coverage_plots
echo "---DONE---"
