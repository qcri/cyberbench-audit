"""Generate report figures from analysis/reports/ CSVs.

Outputs PNGs into analysis/reports/figures/.
Run:
    /export/home/aberriche/miniconda3/envs/seg_zero/bin/python -m analysis.make_plots
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})


def read_csv(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- accuracy heatmap
def plot_accuracy_heatmap():
    rows = read_csv(REPORTS / "results_table.csv")
    models = [c for c in rows[0].keys() if c != "task"]
    tasks = [r["task"] for r in rows]
    M = np.full((len(tasks), len(models)), np.nan)
    for i, r in enumerate(rows):
        for j, m in enumerate(models):
            v = to_float(r[m])
            if v is not None:
                M[i, j] = v

    fig_h = max(4.5, 0.30 * len(tasks) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("-Instruct", "").replace("-70B", "")[:20] for m in models],
                       rotation=40, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    for i in range(len(tasks)):
        for j in range(len(models)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.55 else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Accuracy")
    ax.set_title("Per-(model, sub-task) accuracy")
    fig.savefig(FIGS / "accuracy_heatmap.png")
    plt.close(fig)


def plot_per_model_average():
    rows = read_csv(REPORTS / "results_table.csv")
    models = [c for c in rows[0].keys() if c != "task"]
    avgs = {m: [] for m in models}
    for r in rows:
        for m in models:
            v = to_float(r[m])
            if v is not None:
                avgs[m].append(v)
    means = {m: float(np.mean(avgs[m])) for m in models}
    sorted_m = sorted(means, key=means.get, reverse=True)
    vals = [means[m] for m in sorted_m]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(range(len(sorted_m)), vals, color="#3a7bd5")
    ax.set_yticks(range(len(sorted_m)))
    ax.set_yticklabels(sorted_m)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-average accuracy")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_title("Models ranked by mean accuracy across 24 sub-tasks")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.savefig(FIGS / "per_model_average.png")
    plt.close(fig)


# -------------------------------------------------------------- threshold sweep
def plot_threshold_sweep():
    rows = read_csv(REPORTS / "gold_errors" / "threshold_sweep.csv")
    thresholds = [c for c in rows[0].keys() if c not in {"task", "n_samples"}]
    thr_vals = [float(t) for t in thresholds]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = matplotlib.colormaps["tab20"].resampled(len(rows))
    for i, r in enumerate(rows):
        n = max(int(r["n_samples"]), 1)
        pct = [100 * int(r[t]) / n for t in thresholds]
        ax.plot(thr_vals, pct, marker="o", color=cmap(i),
                label=f"{r['task']} (n={r['n_samples']})", linewidth=1.3, markersize=4)
    ax.set_xlabel("Agreement-fraction threshold")
    ax.set_ylabel("% samples flagged")
    ax.set_title("Threshold sweep — % flagged per sub-task (plain unweighted majority)")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(ncol=2, fontsize=7, loc="upper right")
    fig.savefig(FIGS / "threshold_sweep.png")
    plt.close(fig)


def plot_threshold_aggregate():
    rows = read_csv(REPORTS / "gold_errors" / "threshold_sweep.csv")
    thresholds = [c for c in rows[0].keys() if c not in {"task", "n_samples"}]
    thr_vals = [float(t) for t in thresholds]
    totals = [sum(int(r[t]) for r in rows) for t in thresholds]
    n_total = sum(int(r["n_samples"]) for r in rows)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(thr_vals, totals, "o-", color="#c0392b", linewidth=1.5)
    for x, y in zip(thr_vals, totals):
        ax.annotate(str(y), (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("Agreement-fraction threshold")
    ax.set_ylabel("Total samples flagged (across all sub-tasks)")
    ax.set_title(f"Total flagged samples vs threshold (out of {n_total} votable samples)")
    ax.grid(linestyle=":", alpha=0.5)
    fig.savefig(FIGS / "threshold_total.png")
    plt.close(fig)


# ----------------------------------------------------- weighted comparison plot
def plot_weighted_comparison():
    sweep = {r["task"]: r for r in read_csv(REPORTS / "gold_errors" / "threshold_sweep.csv")}
    weighted = read_csv(REPORTS / "weighted_rank" / "summary.csv")

    tasks = [r["task"] for r in weighted]
    plain = [int(sweep[t]["0.50"]) for t in tasks]
    linear = [int(r["linear"]) for r in weighted]
    harmonic = [int(r["harmonic"]) for r in weighted]
    n_samples = [int(r["n_samples"]) for r in weighted]

    plain_pct = [100 * p / n for p, n in zip(plain, n_samples)]
    linear_pct = [100 * l / n for l, n in zip(linear, n_samples)]
    harm_pct = [100 * h / n for h, n in zip(harmonic, n_samples)]

    x = np.arange(len(tasks))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w, plain_pct, w, label="plain (≥0.50)", color="#3a7bd5")
    ax.bar(x, linear_pct, w, label="linear-weighted", color="#27ae60")
    ax.bar(x + w, harm_pct, w, label="harmonic-weighted", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel("% samples flagged")
    ax.set_title("Plain vs rank-weighted majority (threshold = 0.50)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.savefig(FIGS / "weighted_comparison.png")
    plt.close(fig)


# --------------------------------------------------------------- top-k heatmap
def plot_topk_heatmap():
    rows = read_csv(REPORTS / "topk" / "summary.csv")
    k_cols = [c for c in rows[0].keys() if c.startswith("k=")]
    M = np.zeros((len(rows), len(k_cols)))
    for i, r in enumerate(rows):
        n = max(int(r["n_samples"]), 1)
        for j, k in enumerate(k_cols):
            M[i, j] = 100 * int(r[k]) / n

    fig_h = max(4.5, 0.32 * len(rows) + 1)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    im = ax.imshow(M, aspect="auto", cmap="magma_r")
    ax.set_xticks(range(len(k_cols)))
    ax.set_xticklabels([k.replace("k=", "k=") for k in k_cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["task"] for r in rows])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                    color="white" if M[i, j] > M.max() * 0.6 else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("% flagged")
    ax.set_title("Top-k voting — % flagged at threshold 0.50")
    fig.savefig(FIGS / "topk_heatmap.png")
    plt.close(fig)


# --------------------------------------------------------- acceptance heatmap
def plot_acceptance_heatmap():
    rows = read_csv(REPORTS / "acceptance" / "summary.csv")
    elig = {r["task"]: r for r in read_csv(REPORTS / "acceptance" / "eligible_models.csv")}
    alphas = [c for c in rows[0].keys() if c.startswith("alpha=")]
    M = np.full((len(rows), len(alphas)), np.nan)
    quorum = np.zeros_like(M, dtype=bool)
    for i, r in enumerate(rows):
        n = max(int(r["n_samples"]), 1)
        for j, a in enumerate(alphas):
            v = int(r[a])
            if v == -1:
                quorum[i, j] = True
            else:
                M[i, j] = 100 * v / n

    fig_h = max(4.5, 0.32 * len(rows) + 1)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    im = ax.imshow(M, aspect="auto", cmap="cividis")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([a.replace("alpha=", "α=") for a in alphas])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["task"] for r in rows])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if quorum[i, j]:
                ax.text(j, i, "n/a", ha="center", va="center", color="#666", fontsize=7)
            elif not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        color="white" if M[i, j] > np.nanmax(M) * 0.55 else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("% flagged")
    ax.set_title("Acceptance-cutoff voting — % flagged (n/a = <3 eligible models)")
    fig.savefig(FIGS / "acceptance_heatmap.png")
    plt.close(fig)


def plot_eligible_count():
    rows = read_csv(REPORTS / "acceptance" / "eligible_models.csv")
    alphas = [c for c in rows[0].keys() if c.startswith("alpha=")]
    M = np.zeros((len(rows), len(alphas)))
    for i, r in enumerate(rows):
        for j, a in enumerate(alphas):
            M[i, j] = int(r[a])

    fig_h = max(4.5, 0.32 * len(rows) + 1)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    im = ax.imshow(M, aspect="auto", cmap="Greens", vmin=0, vmax=10)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([a.replace("alpha=", "α=") for a in alphas])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["task"] for r in rows])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center",
                    color="black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("# eligible models")
    ax.set_title("Number of models clearing the acceptance cutoff (cutoff = best · α)")
    fig.savefig(FIGS / "eligible_count.png")
    plt.close(fig)


def main():
    plot_accuracy_heatmap()
    plot_per_model_average()
    plot_threshold_sweep()
    plot_threshold_aggregate()
    plot_weighted_comparison()
    plot_topk_heatmap()
    plot_acceptance_heatmap()
    plot_eligible_count()
    print(f"Figures written to {FIGS}/")


if __name__ == "__main__":
    main()
