# Analysis pipeline

Re-runnable analysis on top of `outputs/`:

0. **`judge_agreement.py`** — 3-judge agreement report (default vs v1 vs v2).
   Default and v1 judge the same `responses_<MODEL>/` set so we report
   per-sample Cohen's κ + raw agreement; v2 judges a different inference run
   so we only report aggregate cell accuracy for it. Outputs land in
   `analysis/reports/judge_agreement/`.
1. **`results_table.py`** — master accuracy table (24 sub-tasks: was 22, plus
   `cti_taa` and `sevenllm`). Each cell is a per-sample **majority vote** over
   every available judge run (default, _v1, _v2): aligned by sample `index`,
   majority taken, then accuracy = mean(majority_correct). Skipped samples
   (judge API failures) are excluded from the vote. The meta CSV's source
   column (`def`, `v1`, `v2`, `def+v1`, `def+v1+v2`, …) records which judges
   contributed to each cell.
2. **`gold_error_voting.py`** — flag samples with suspect gold answers via majority voting across model predictions on votable sub-tasks (MCQ, ID, VSP, TFX). Free-form tasks (`rcm`, `taa`, `athena_rcm`) are skipped.

## Run

```bash
cd BenchmarkingSecBenchmarks
PYTHONPATH=. python3 -m analysis.judge_agreement
PYTHONPATH=. python3 -m analysis.results_table
PYTHONPATH=. python3 -m analysis.gold_error_voting           # full run
PYTHONPATH=. python3 -m analysis.gold_error_voting --aggregate  # only re-aggregate partial JSONs
```

For the first two steps, large I/O can trip login-node guards — submit via
slurm (cpu-all) instead:

```bash
sbatch --export=ALL,TARGETS=judge_agreement slurm/run_analysis.sh
sbatch --export=ALL,TARGETS=results_table slurm/run_analysis.sh
# Or both in one job (default):
sbatch slurm/run_analysis.sh
```

If your environment time-limits the python process and a single task is too slow:

```bash
# 1) Cache the prediction matrix incrementally (resumable across calls).
PYTHONPATH=. python3 -m analysis.gold_error_voting --prepare-matrix redsage_skills

# 2) Process voting in chunks (matrix is loaded from cache).
for i in 0 1 2 3; do
  PYTHONPATH=. python3 -m analysis.gold_error_voting \
    --task redsage_skills --chunk-idx $i --n-chunks 4
done

# 3) Merge chunks into the per-task partial.
PYTHONPATH=. python3 -m analysis.gold_error_voting --merge-chunks redsage_skills

# 4) Aggregate every per-task partial into final CSVs/MD.
PYTHONPATH=. python3 -m analysis.gold_error_voting --aggregate
```

## Outputs (under `analysis/reports/`)

- `judge_agreement/per_cell_default_vs_v1.csv` — κ, raw agreement, # disagreements per (model, task).
- `judge_agreement/per_cell_aggregate_compare.csv` — accuracy(default/v1/v2) per (model, task).
- `judge_agreement/summary.md` — overall κ averages, per-task and per-model breakdowns, lowest-κ cells.
- `judge_agreement/disagreements/<model>__<task>.jsonl` — sample-level rows where default and v1 disagree (capped at 25/cell).
- `results_table.csv` / `.md` / `_meta.csv` — master accuracy table + per-cell source label.
- `per_model_task_accuracy.json` — cached map consumed by Step 2.
- `gold_errors/threshold_sweep.{csv,md}` + `flagged/<task>_t50.jsonl` — plain unweighted majority across thresholds 0.30 → 1.00.
- `weighted_rank/summary.csv` + `flagged/<task>_w<scheme>.jsonl` — rank-weighted vote with `linear` and `harmonic` weighting.
- `topk/summary.csv` + `flagged/<task>_k<k>.jsonl` — only top-k models (by per-task accuracy) participate.
- `acceptance/summary.csv` + `eligible_models.csv` + `flagged/<task>_a<alpha>.jsonl` — only models with accuracy ≥ `best * alpha` participate; `-1` means fewer than 3 eligible models (quorum unmet).
