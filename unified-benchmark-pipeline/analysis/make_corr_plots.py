"""Figures for the cross-task correlation section. NeurIPS-quality output.

Inputs from analysis/reports/correlation/.
Outputs PNGs to analysis/reports/figures/ at 300 DPI.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.cluster import hierarchy

from analysis.lib.loaders import PARENT_GROUPS
from analysis.lib.plot_style import (
    PALETTE,
    PARENT_COLOURS,
    add_cluster_boxes,
    apply_style,
    colour_xticklabels_by_parent,
    light_grid,
    parent_colour,
    parent_legend_handles,
    parent_of,
    square_heatmap_aspect,
    thin_spines,
)


HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
CORR = REPORTS / "correlation"
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

apply_style()


def load_kendall_matrix():
    rows = list(csv.DictReader(open(CORR / "pairwise_kendall.csv")))
    tasks = [r["task"] for r in rows]
    n = len(tasks)
    K = np.zeros((n, n))
    for i, r in enumerate(rows):
        for j, t in enumerate(tasks):
            v = r[t]
            K[i, j] = float(v) if v else np.nan
    return tasks, K


def load_clusters():
    return json.loads((CORR / "clusters.json").read_text())


def load_linkage():
    rows = list(csv.DictReader(open(CORR / "linkage.csv")))
    Z = np.array([[float(r["id1"]), float(r["id2"]), float(r["distance"]), float(r["n"])]
                  for r in rows])
    return Z


def load_pairs():
    return list(csv.DictReader(open(CORR / "bootstrap_ci.csv")))


def load_pca_proj():
    return list(csv.DictReader(open(CORR / "pca_projection.csv")))


def load_pca_components():
    return list(csv.DictReader(open(CORR / "pca_components.csv")))


def load_eff():
    return json.loads((CORR / "effective_dimensions.json").read_text())


# -------------------- 1. Kendall heatmap --------------------

def plot_kendall_heatmap_clustered():
    tasks, K = load_kendall_matrix()
    cl = load_clusters()
    order = cl["leaf_order"]
    cluster_at_5 = [cl["clusters"]["5"][i] for i in order]
    K2 = K[np.ix_(order, order)]
    labels = [tasks[i] for i in order]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    im = ax.imshow(K2, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal",
                   interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7,
                       rotation_mode="anchor")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    colour_xticklabels_by_parent(ax, labels, PARENT_GROUPS)

    # Highlight cluster blocks discussed in main text.
    add_cluster_boxes(ax, cluster_at_5, lw=1.4, color="0.18")

    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02, shrink=0.92)
    cbar.set_label(r"Kendall $\tau$-b", fontsize=9)
    cbar.ax.tick_params(labelsize=7.5)

    # Parent-colour legend (small, top-right outside).
    handles = parent_legend_handles(PARENT_GROUPS)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.18, 1.0),
              fontsize=7, frameon=False, title="parent", title_fontsize=8)

    fig.savefig(FIGS / "kendall_heatmap_clustered.png", dpi=300)
    plt.close(fig)


# -------------------- 2. Dendrogram --------------------

def plot_dendrogram():
    Z = load_linkage()
    cl = load_clusters()
    tasks = cl["tasks"]

    # cut threshold for k=5 partitions.
    n = len(tasks)
    k = 5
    if Z.shape[0] >= n - k:
        cut_threshold = Z[n - k - 1, 2] - 1e-6
    else:
        cut_threshold = 0.7

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    hierarchy.set_link_color_palette([
        PALETTE["blue"], PALETTE["green"], PALETTE["red"],
        PALETTE["orange"], PALETTE["purple"], PALETTE["brown"],
    ])
    hierarchy.dendrogram(
        Z,
        labels=tasks,
        leaf_rotation=55,
        leaf_font_size=7.5,
        color_threshold=cut_threshold,
        above_threshold_color="0.55",
        ax=ax,
    )
    # Tint leaf labels by parent.
    for t in ax.get_xticklabels():
        t.set_color(parent_colour(t.get_text(), PARENT_GROUPS))
        t.set_horizontalalignment("right")
        t.set_rotation_mode("anchor")

    ax.axhline(cut_threshold, color="0.45", linewidth=0.7, linestyle="--")
    ax.text(0.99, cut_threshold, f"  k={k} cut",
            transform=ax.get_yaxis_transform(), va="bottom", ha="right",
            fontsize=7.5, color="0.30")
    ax.set_ylabel(r"Distance ($1{-}\tau$)")
    light_grid(ax, axis="y")
    thin_spines(ax)

    handles = parent_legend_handles(PARENT_GROUPS)
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              frameon=False, title="parent", title_fontsize=8, ncol=2)

    fig.savefig(FIGS / "dendrogram.png", dpi=300)
    plt.close(fig)


# -------------------- 3. PCA scree --------------------

def plot_scree():
    eff = load_eff()
    ratio = np.array(eff["explained_variance_ratio"])
    null95 = np.array(eff["null_variance_ratio_p95"])
    cum = np.array(eff["cumulative_variance"])
    n = len(ratio)
    x = np.arange(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2))

    # Left: scree (all components, log-y to surface PC2..n)
    ax1.bar(x, ratio * 100, color=PALETTE["blue"], edgecolor="0.2",
            linewidth=0.5, label="observed")
    ax1.plot(x, null95 * 100, "x--", color=PALETTE["red"], linewidth=1.0,
             markersize=5.5, label="Horn null (95th pct)")
    # Annotate PC1 value.
    ax1.text(1, ratio[0] * 100, f"  {ratio[0]*100:.1f}%",
             va="center", ha="left", fontsize=8, color="0.10")
    ax1.set_xticks(x)
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_yticks([0, 1, 5, 10, 50, 100])
    ax1.set_yticklabels(["0", "1", "5", "10", "50", "100"])
    ax1.set_ylim(0, 110)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("% variance explained")
    ax1.legend(loc="upper right", fontsize=8)
    light_grid(ax1, axis="y")
    thin_spines(ax1)

    # Right: cumulative variance with full y range
    ax2.plot(x, cum * 100, "o-", color=PALETTE["green"], linewidth=1.4,
             markersize=4.5)
    ax2.axhline(90, color="0.55", linewidth=0.7, linestyle="--")
    ax2.text(n, 90, "  90%", va="center", ha="right", fontsize=7.5, color="0.40")
    ax2.annotate(f"PC1 = {cum[0]*100:.1f}%",
                 xy=(1, cum[0] * 100), xytext=(2.5, 70),
                 fontsize=8, color="0.10",
                 arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7))
    ax2.set_xticks(x)
    ax2.set_ylim(0, 102)
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Cumulative % variance")
    light_grid(ax2, axis="y")
    thin_spines(ax2)

    fig.savefig(FIGS / "pca_scree.png", dpi=300)
    plt.close(fig)


# -------------------- 4. PCA biplot --------------------

def plot_pca_biplot():
    rows = load_pca_proj()
    tasks = [r["task"] for r in rows]
    pc1 = np.array([float(r["PC1"]) for r in rows])
    pc2 = np.array([float(r["PC2"]) for r in rows])

    cl = load_clusters()
    cluster_at_5 = cl["clusters"]["5"]
    n_clusters = max(cluster_at_5)

    cluster_palette = {
        1: PALETTE["red"],     # floor-effect
        2: PALETTE["green"],   # reasoning
        3: PALETTE["blue"],    # dominant
        4: PALETTE["orange"],  # cti_taa singleton
        5: PALETTE["purple"],  # secure_kcv singleton
    }
    cluster_label = {
        1: "floor-effect (vsp/rms/taa)",
        2: "reasoning",
        3: "dominant block",
        4: "cti_taa (singleton)",
        5: "secure_kcv (singleton)",
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for c in range(1, n_clusters + 1):
        idx = [i for i, x in enumerate(cluster_at_5) if x == c]
        ax.scatter(pc1[idx], pc2[idx], s=85,
                   color=cluster_palette.get(c, PALETTE["grey"]),
                   edgecolor="white", linewidth=1.2,
                   label=cluster_label.get(c, f"cluster {c}"), zorder=3)

    # Manual offsets to spread the dense right-side dominant cluster.
    overrides = {
        "redsage_kali":      (-2, 8),
        "redsage_skills":    (-30, 16),
        "redsage_cli":       (-30, -10),
        "redsage_frameworks":(8,  16),
        "redsage_generals":  (8,  6),
        "secure_maet":       (8, -8),
        "secure_cwet":       (8, -16),
        "secbench":          (-32, -10),
        "mcq":               (-12, -12),
        "ckt":               (8, 8),
        "cybermetric":       (8,  18),
        "mmlu_cs":           (-50, 0),
        "athena_vsp":        (8, 0),
        "ate":               (8, 4),
        "athena_ate":        (8, 0),
        "athena_rcm":        (8, 0),
        "rcm":               (8, 0),
        "sevenllm":          (8, 6),
        "secure_kcv":        (8, 0),
        "cti_taa":           (8, -4),
        "vsp":               (8, 0),
        "rms":               (8, 0),
        "taa":               (8, -4),
        "seceval":           (-44, 0),
    }
    for i, t in enumerate(tasks):
        dx, dy = overrides.get(t, (6, 4))
        ax.annotate(t, (pc1[i], pc2[i]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=7, color="0.10")

    eff = load_eff()
    ax.set_xlabel(f"PC1 ({eff['explained_variance_ratio'][0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({eff['explained_variance_ratio'][1]*100:.1f}%)")
    ax.axhline(0, color="0.6", linewidth=0.5, zorder=1)
    ax.axvline(0, color="0.6", linewidth=0.5, zorder=1)
    light_grid(ax)
    thin_spines(ax)
    ax.legend(loc="lower left", fontsize=7.5, frameon=True)

    fig.savefig(FIGS / "pca_biplot.png", dpi=300)
    plt.close(fig)


# -------------------- 5. Within vs across parent --------------------

def plot_within_vs_across():
    pairs = load_pairs()
    within = [float(p["kendall"]) for p in pairs if p["same_parent"] == "True"]
    across = [float(p["kendall"]) for p in pairs if p["same_parent"] == "False"]
    n_w, n_a = len(within), len(across)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bp = ax.boxplot([within, across],
                    tick_labels=[f"within\n(n={n_w})", f"across\n(n={n_a})"],
                    patch_artist=True, widths=0.45,
                    showmeans=True,
                    meanprops=dict(marker="D", markersize=6,
                                   markerfacecolor="white",
                                   markeredgecolor="0.10", markeredgewidth=1.0),
                    medianprops=dict(color="0.10", linewidth=1.2),
                    flierprops=dict(marker=".", markerfacecolor="0.4",
                                    markeredgecolor="0.4", markersize=3))
    for patch, color in zip(bp["boxes"], [PALETTE["blue"], PALETTE["red"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor("0.20")
        patch.set_linewidth(0.8)

    rng = np.random.default_rng(0)
    ax.scatter(1 + rng.uniform(-0.10, 0.10, n_w), within,
               color=PALETTE["blue"], alpha=0.55, s=14, zorder=3,
               edgecolor="white", linewidth=0.3)
    ax.scatter(2 + rng.uniform(-0.10, 0.10, n_a), across,
               color=PALETTE["red"], alpha=0.45, s=12, zorder=3,
               edgecolor="white", linewidth=0.3)

    ax.set_ylabel(r"Kendall $\tau$")
    ax.set_ylim(-0.55, 1.05)
    ax.axhline(0, color="0.55", linewidth=0.5, zorder=1)
    light_grid(ax, axis="y")
    thin_spines(ax)

    mu_w, mu_a = float(np.mean(within)), float(np.mean(across))
    ax.text(0.98, 0.05,
            (f"$\\mu_{{\\mathrm{{within}}}} = {mu_w:.4f}$\n"
             f"$\\mu_{{\\mathrm{{across}}}} = {mu_a:.4f}$\n"
             r"$10^4$-perm $p = 0.998$"),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="0.10",
            bbox=dict(facecolor="white", edgecolor="0.7",
                      boxstyle="round,pad=0.35", linewidth=0.6))

    mean_handle = plt.Line2D([0], [0], marker="D", color="w",
                             markerfacecolor="white", markeredgecolor="0.10",
                             markersize=6, label="mean")
    median_handle = plt.Line2D([0], [0], color="0.10", linewidth=1.2, label="median")
    ax.legend(handles=[mean_handle, median_handle], loc="upper right",
              fontsize=7.5, frameon=True)

    fig.savefig(FIGS / "within_vs_across_parent.png", dpi=300)
    plt.close(fig)


# -------------------- 6. Top correlated pairs --------------------

def plot_top_pairs():
    pairs = load_pairs()
    sorted_pairs = sorted(pairs, key=lambda p: -float(p["kendall"]))[:15]
    labels = [f"{p['task_a']} ↔ {p['task_b']}" for p in sorted_pairs]
    tau = np.array([float(p["kendall"]) for p in sorted_pairs])
    lo = np.array([float(p["ci_lo"]) for p in sorted_pairs])
    hi = np.array([float(p["ci_hi"]) for p in sorted_pairs])
    same = [p["same_parent"] == "True" for p in sorted_pairs]
    colours = [PALETTE["blue"] if s else PALETTE["red"] for s in same]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    y = np.arange(len(labels))
    for yi, t, l, h, c in zip(y, tau, lo, hi, colours):
        ax.errorbar(t, yi, xerr=[[t - l], [h - t]],
                    fmt="o", color=c, ecolor="0.55",
                    capsize=2.5, markersize=5.5, linewidth=1.0,
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color="0.55", linewidth=0.5)
    ax.axvline(0.85, color=PALETTE["orange"], linewidth=0.8, linestyle="--")
    ax.text(0.85, -0.6, r" $\tau{=}0.85$",
            color=PALETTE["orange"], fontsize=7.5, va="top")
    ax.set_xlabel(r"Kendall $\tau$ (point estimate, bootstrap 95% CI)")
    ax.set_xlim(-0.05, 1.05)
    light_grid(ax, axis="x")
    thin_spines(ax)

    h_within = plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=PALETTE["blue"],
                          markeredgecolor="white", markersize=6, label="within parent")
    h_across = plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=PALETTE["red"],
                          markeredgecolor="white", markersize=6, label="across parent")
    ax.legend(handles=[h_within, h_across], loc="lower right",
              fontsize=7.5, frameon=True)

    fig.savefig(FIGS / "top_correlated_pairs.png", dpi=300)
    plt.close(fig)


# -------------------- main --------------------

def main():
    plot_kendall_heatmap_clustered()
    plot_dendrogram()
    plot_scree()
    plot_pca_biplot()
    plot_within_vs_across()
    plot_top_pairs()
    print(f"figures written to {FIGS}/")


if __name__ == "__main__":
    main()
