"""Shared NeurIPS-quality plotting style for `make_*_plots.py` scripts.

Usage:
    from analysis.lib.plot_style import apply_style, PALETTE, parent_colour
    apply_style()
    fig, ax = plt.subplots(figsize=...)
    ...
    fig.savefig(path, dpi=300, bbox_inches="tight")

Design choices (defaults targeting NeurIPS print column at 6.5" wide):
  - savefig DPI 300 + bbox_inches='tight' (set by apply_style; per-plot can override)
  - serif font (DejaVu Serif via matplotlib bundled; falls back gracefully)
  - light-weight grids only on axes that need them
  - colourblind-safe categorical palette (Tableau 10 / Set1 mix)
  - sequential = viridis; diverging = RdBu_r
  - axis spines kept (matches NeurIPS journal style); top/right hidden only
    when explicitly requested via the helpers
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ----- shared parameters ---------------------------------------------------

# Categorical palette: colourblind-safe (CUD + Tableau-10 selection).
PALETTE = {
    "blue":     "#1f77b4",
    "orange":   "#ff7f0e",
    "green":    "#2ca02c",
    "red":      "#d62728",
    "purple":   "#9467bd",
    "brown":    "#8c564b",
    "pink":     "#e377c2",
    "grey":     "#7f7f7f",
    "olive":    "#bcbd22",
    "cyan":     "#17becf",
}

# Categorical class colours used across the suite.
KA_COLOURS = {
    "K":         "#3a7bd5",
    "A":         "#c0392b",
    "ambiguous": "#f1c232",
}

# Parent-benchmark colours (used by coverage_appendix and overlap heatmaps).
PARENT_COLOURS = {
    "CTI":            "#1f77b4",
    "ATHENA":         "#9467bd",
    "SECURE":         "#2ca02c",
    "REDSAGE":        "#d62728",
    "CYBERMETRIC":    "#ff7f0e",
    "MCQ-Standalone": "#8c564b",
    "SEVENLLM":       "#17becf",
}

# Quadrant palette (semantic vs accuracy scatter).
QUADRANT_COLOURS = {
    "redundant":         "#c0392b",
    "general-axis-only": "#1f77b4",
    "format-quirk":      "#f39c12",
    "diversifying":      "#7f8c8d",
}


def parent_of(task: str, parent_groups) -> str:
    for parent, members in parent_groups:
        if task in members:
            return parent
    return "OTHER"


def parent_colour(task: str, parent_groups) -> str:
    return PARENT_COLOURS.get(parent_of(task, parent_groups), "#444")


# ----- the global style ----------------------------------------------------

_RC = {
    # geometry / output
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.06,
    # fonts
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset":   "stix",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9.5,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.frameon":     True,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "0.85",
    # axes / grid
    "axes.linewidth":     0.8,
    "axes.edgecolor":     "0.25",
    "axes.grid":          False,            # turn on locally if needed
    "grid.alpha":         0.3,
    "grid.linestyle":     ":",
    "grid.linewidth":     0.6,
    # ticks (inward, thin)
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
}


_APPLIED = False


def apply_style():
    """Idempotent global style application."""
    global _APPLIED
    if _APPLIED:
        return
    plt.rcParams.update(_RC)
    _APPLIED = True


# ----- helpers used by multiple plot scripts -------------------------------

def square_heatmap_aspect(ax):
    """Force square cells on a heatmap (default for n×n matrices)."""
    ax.set_aspect("equal", adjustable="box")


def add_cluster_boxes(ax, cluster_assignments, *, lw=1.4, color="#222"):
    """Draw rectangle borders around contiguous-cluster blocks on a heatmap.

    `cluster_assignments` is a list of cluster-id ints in the row/column order
    already used by the heatmap (after leaf-order reshuffling).
    """
    n = len(cluster_assignments)
    start = 0
    for i in range(1, n + 1):
        if i == n or cluster_assignments[i] != cluster_assignments[start]:
            size = i - start
            rect = mpatches.Rectangle(
                (start - 0.5, start - 0.5), size, size,
                linewidth=lw, edgecolor=color, facecolor="none",
                joinstyle="miter",
            )
            ax.add_patch(rect)
            start = i


def colour_xticklabels_by_parent(ax, labels, parent_groups):
    """Tint each tick label by its parent benchmark."""
    for tick, lab in zip(ax.get_xticklabels(), labels):
        tick.set_color(parent_colour(lab, parent_groups))
    for tick, lab in zip(ax.get_yticklabels(), labels):
        tick.set_color(parent_colour(lab, parent_groups))


def light_grid(ax, axis="both"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, linestyle=":", linewidth=0.6, alpha=0.35, color="0.5")


def thin_spines(ax, sides=("top", "right")):
    for s in sides:
        ax.spines[s].set_visible(False)


def parent_legend_handles(parent_groups):
    """Return matplotlib handles for a parent-colour legend."""
    return [
        mpatches.Patch(facecolor=PARENT_COLOURS[p], edgecolor="0.3", linewidth=0.5, label=p)
        for p, _ in parent_groups if p in PARENT_COLOURS
    ]


def annotate_with_offset_avoidance(ax, labels, xs, ys, *, fontsize=6.5,
                                   max_iters=200, padx=0.012, pady=0.018):
    """Greedy O(n^2) point-label de-overlap: nudges each label away from the
    closest already-placed label until no two boxes overlap.

    Designed for small N (<=30 sub-tasks). For larger N use adjustText.
    """
    placed = []  # (x, y, dx, dy)
    xrange = (max(xs) - min(xs)) or 1.0
    yrange = (max(ys) - min(ys)) or 1.0
    px = padx * xrange
    py = pady * yrange
    for x, y, lab in zip(xs, ys, labels):
        dx, dy = 0.012 * xrange, 0.018 * yrange
        for _ in range(max_iters):
            collide = False
            for px_, py_, pdx, pdy in placed:
                if abs((x + dx) - (px_ + pdx)) < px and abs((y + dy) - (py_ + pdy)) < py:
                    collide = True
                    break
            if not collide:
                break
            dy += py * 0.6  # bump upward
        placed.append((x, y, dx, dy))
        ax.annotate(lab, (x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=fontsize, color="#111",
                    arrowprops=None)
