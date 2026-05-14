"""Figures for the embedding-based redundancy section. NeurIPS-quality output."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from analysis.lib.loaders import PARENT_GROUPS
from analysis.lib.plot_style import (
    PALETTE,
    PARENT_COLOURS,
    QUADRANT_COLOURS,
    add_cluster_boxes,
    apply_style,
    colour_xticklabels_by_parent,
    light_grid,
    parent_colour,
    parent_legend_handles,
    thin_spines,
)


HERE = Path(__file__).resolve().parent
EMBED = HERE / "reports" / "embeddings"
CORR = HERE / "reports" / "correlation"
FIGS = HERE / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

apply_style()


def load_matrix_csv(path):
    rows = list(csv.DictReader(open(path)))
    tasks = [r["task"] for r in rows]
    n = len(tasks)
    M = np.full((n, n), np.nan)
    for i, r in enumerate(rows):
        for j, t in enumerate(tasks):
            v = r[t]
            if v != "":
                M[i, j] = float(v)
    return tasks, M


def load_clusters_leaf_order():
    p = CORR / "clusters.json"
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    return d.get("tasks"), d.get("leaf_order")


def _load_cluster_at_5(tasks_in_order):
    """Return the k=5 cluster id for each task in the supplied order."""
    p = CORR / "clusters.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    base_tasks = d["tasks"]
    cl5 = d["clusters"]["5"]
    name_to_cluster = {t: c for t, c in zip(base_tasks, cl5)}
    return [name_to_cluster.get(t, 0) for t in tasks_in_order]


# ---------------- 1. centroid heatmap ----------------

def plot_centroid_heatmap():
    tasks, S = load_matrix_csv(EMBED / "centroid_similarity.csv")
    leaf_tasks, leaf_order = load_clusters_leaf_order()
    if leaf_tasks and leaf_order:
        order = []
        leaf_seq = [leaf_tasks[i] for i in leaf_order]
        for t in leaf_seq:
            if t in tasks:
                order.append(tasks.index(t))
        for i in range(len(tasks)):
            if i not in order:
                order.append(i)
    else:
        order = list(range(len(tasks)))
    S2 = S[np.ix_(order, order)]
    labels = [tasks[i] for i in order]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    im = ax.imshow(S2, cmap="viridis", vmin=0.4, vmax=1.0,
                   aspect="equal", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7,
                       rotation_mode="anchor")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    colour_xticklabels_by_parent(ax, labels, PARENT_GROUPS)

    # Highlight cells > 0.95 (excluding diagonal): paper references this.
    high = (S2 > 0.95) & ~np.eye(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if high[i, j]:
                rect = mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=1.0, edgecolor=PALETTE["red"],
                    facecolor="none",
                )
                ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02, shrink=0.92)
    cbar.set_label("cosine similarity", fontsize=9)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.ax.axhline(0.95, color=PALETTE["red"], linewidth=0.8)

    handles = parent_legend_handles(PARENT_GROUPS) + [
        mpatches.Patch(facecolor="none", edgecolor=PALETTE["red"],
                       linewidth=1.0, label="cos > 0.95")
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.18, 1.0),
              fontsize=7, frameon=False)

    fig.savefig(FIGS / "centroid_similarity_heatmap.png", dpi=300)
    plt.close(fig)


# ---------------- 2. semantic vs accuracy scatter ----------------

def plot_semantic_vs_accuracy():
    rows = list(csv.DictReader(open(EMBED / "centroid_vs_kendall.csv")))
    sim = np.array([float(r["semantic_sim"]) for r in rows])
    tau = np.array([float(r["kendall_tau"]) if r["kendall_tau"] else np.nan
                    for r in rows])
    quad = np.array([r["quadrant"] for r in rows])
    mask = ~np.isnan(tau)
    sim, tau, quad = sim[mask], tau[mask], quad[mask]
    rows = [r for r, m in zip(rows, mask) if m]
    n_total = len(rows)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    counts = {q: int((quad == q).sum()) for q in QUADRANT_COLOURS}
    for q, c in QUADRANT_COLOURS.items():
        m = quad == q
        ax.scatter(sim[m], tau[m], s=24, color=c,
                   label=f"{q} (n={counts[q]})",
                   alpha=0.78, edgecolor="white", linewidth=0.4, zorder=3)

    by_quad = {q: [] for q in QUADRANT_COLOURS}
    for r in rows:
        by_quad[r["quadrant"]].append(r)

    # Annotate up to 4 most-extreme pairs in each quadrant of interest.
    for r in sorted(by_quad.get("redundant", []),
                    key=lambda r: -float(r["kendall_tau"]))[:3]:
        ax.annotate(f"{r['task_a']}↔{r['task_b']}",
                    (float(r["semantic_sim"]), float(r["kendall_tau"])),
                    xytext=(4, 4), textcoords="offset points", fontsize=6.5,
                    color="0.10")
    for r in sorted(by_quad.get("format-quirk", []),
                    key=lambda r: -float(r["semantic_sim"]))[:3]:
        ax.annotate(f"{r['task_a']}↔{r['task_b']}",
                    (float(r["semantic_sim"]), float(r["kendall_tau"])),
                    xytext=(4, -10), textcoords="offset points", fontsize=6.5,
                    color="0.10")

    ax.axvline(0.85, color="0.55", linewidth=0.6, linestyle="--", zorder=1)
    ax.axhline(0.5, color="0.55", linewidth=0.6, linestyle="--", zorder=1)
    ax.axhline(-0.5, color="0.55", linewidth=0.6, linestyle="--", zorder=1)
    ax.set_xlabel("semantic similarity (centroid cosine)")
    ax.set_ylabel(r"Kendall $\tau$ (model-accuracy ranking)")
    ax.set_xlim(0.55, 1.02)
    light_grid(ax)
    thin_spines(ax)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True)

    mp = EMBED / "mantel.json"
    if mp.exists():
        d = json.loads(mp.read_text())
        ax.text(0.02, 0.98,
                (f"Mantel $r = {d['pearson_r']:.3f}$  "
                 f"($p < 0.002$, {d['n_perm']} perms)\n"
                 f"$n_\\mathrm{{pairs}} = {n_total}$"),
                transform=ax.transAxes, verticalalignment="top",
                fontsize=8, color="0.10",
                bbox=dict(facecolor="white", edgecolor="0.7",
                          boxstyle="round,pad=0.35", linewidth=0.6))

    fig.savefig(FIGS / "semantic_vs_accuracy_scatter.png", dpi=300)
    plt.close(fig)


# ---------------- 3. cluster size distribution ----------------

def plot_evoc_size_distribution():
    rows = list(csv.DictReader(open(EMBED / "cluster_task_purity.csv")))
    sizes = sorted([int(r["size"]) for r in rows], reverse=True)
    if not sizes:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.bar(range(len(sizes)), sizes, color=PALETTE["blue"],
           edgecolor="0.2", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("cluster rank (descending size)")
    ax.set_ylabel("cluster size (log)")
    ax.set_title(f"EVoC cluster size distribution (n_clusters = {len(sizes)})")
    light_grid(ax, axis="y")
    thin_spines(ax)
    fig.savefig(FIGS / "evoc_size_distribution.png", dpi=300)
    plt.close(fig)


# ---------------- 4. cluster composition (top-N) ----------------

def plot_cluster_composition(top_n=40):
    label_rows = list(csv.DictReader(open(EMBED / "evoc_labels.csv")))
    by_cluster = {}
    for r in label_rows:
        c = int(r["cluster"])
        if c < 0:
            continue
        by_cluster.setdefault(c, []).append(r["task"])

    sized = sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))[:top_n]
    tasks_seen = sorted({t for _, ts in sized for t in ts})
    cmap = matplotlib.colormaps["tab20"].resampled(max(20, len(tasks_seen)))
    colour = {t: cmap(i % cmap.N) for i, t in enumerate(tasks_seen)}

    fig, ax = plt.subplots(figsize=(11, 5))
    for x, (_, ts) in enumerate(sized):
        ctr = Counter(ts)
        bottom = 0
        for task, n in ctr.most_common():
            ax.bar(x, n, bottom=bottom, color=colour[task],
                   edgecolor="white", linewidth=0.3)
            bottom += n
    ax.set_xlabel(f"top-{top_n} clusters by size")
    ax.set_ylabel("# questions")
    ax.set_title(f"cluster composition by source sub-task (top {top_n} EVoC clusters)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colour[t]) for t in tasks_seen]
    ax.legend(handles, tasks_seen, ncol=4, fontsize=6.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.10))
    thin_spines(ax)
    fig.savefig(FIGS / "cluster_task_composition.png", dpi=300)
    plt.close(fig)


# ---------------- 5. task overlap heatmap ----------------

def plot_task_overlap_heatmap():
    p = EMBED / "task_overlap_matrix.csv"
    if not p.exists():
        return
    tasks, M = load_matrix_csv(p)
    leaf_tasks, leaf_order = load_clusters_leaf_order()
    if leaf_tasks and leaf_order:
        order = []
        leaf_seq = [leaf_tasks[i] for i in leaf_order]
        for t in leaf_seq:
            if t in tasks:
                order.append(tasks.index(t))
        for i in range(len(tasks)):
            if i not in order:
                order.append(i)
    else:
        order = list(range(len(tasks)))
    M2 = M[np.ix_(order, order)]
    labels = [tasks[i] for i in order]
    n = len(labels)

    # Symmetrize for visualization (mean of A->B and B->A).
    Msym = (M2 + M2.T) / 2.0
    np.fill_diagonal(Msym, 1.0)

    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    # Log-like normalisation so small overlaps are visible.
    from matplotlib.colors import PowerNorm
    im = ax.imshow(Msym, cmap="rocket_r" if "rocket_r" in plt.colormaps()
                   else "magma_r",
                   norm=PowerNorm(gamma=0.45, vmin=0, vmax=1),
                   aspect="equal", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7,
                       rotation_mode="anchor")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    colour_xticklabels_by_parent(ax, labels, PARENT_GROUPS)

    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02, shrink=0.92)
    cbar.set_label("EVoC shared-cluster ratio (symmetric, $\\gamma$=0.45)",
                   fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    fig.savefig(FIGS / "task_overlap_heatmap.png", dpi=300)
    plt.close(fig)


def main():
    plot_centroid_heatmap()
    plot_semantic_vs_accuracy()
    plot_evoc_size_distribution()
    plot_cluster_composition()
    plot_task_overlap_heatmap()
    print(f"figures written to {FIGS}/")


if __name__ == "__main__":
    main()
