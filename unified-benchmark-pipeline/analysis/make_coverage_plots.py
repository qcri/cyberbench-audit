"""Coverage figures for the §3 main paper + appendix. NeurIPS-quality output.

Main figure:
  (a) per-parent K/A stacked bar with percentage labels
  (b) PCA scree on the 24x10 strict-verdict matrix vs Horn's null
Appendix figure: per-sub-task K/A stacked bar (22 originally-classified bars).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.lib.loaders import PARENT_GROUPS
from analysis.lib.plot_style import (
    KA_COLOURS,
    PALETTE,
    apply_style,
    light_grid,
    parent_colour,
    thin_spines,
)


HERE = Path(__file__).resolve().parent
COVERAGE = HERE / "reports" / "coverage"
CORR = HERE / "reports" / "correlation"
FIGS = HERE / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

apply_style()


def _stacked_with_pct(ax, labels, K, A, amb, *, totals=None,
                      bar_width=0.7, show_pct=True, pct_fontsize=7,
                      n_label=True):
    """Draw stacked K/A/ambiguous bars with optional percentage labels."""
    x = np.arange(len(labels))
    K = np.asarray(K, dtype=float)
    A = np.asarray(A, dtype=float)
    amb = np.asarray(amb, dtype=float)
    total = K + A + amb
    bottoms = np.zeros(len(x))

    for vals, name, key in [(K, "K (knowledge)", "K"),
                             (A, "A (analytical)", "A"),
                             (amb, "ambiguous", "ambiguous")]:
        ax.bar(x, vals, bar_width, bottom=bottoms, label=name,
               color=KA_COLOURS[key], edgecolor="white", linewidth=0.4)
        if show_pct:
            for i, (v, b) in enumerate(zip(vals, bottoms)):
                if total[i] == 0 or v / total[i] < 0.08:
                    continue
                ax.text(x[i], b + v / 2, f"{100*v/total[i]:.0f}%",
                        ha="center", va="center", fontsize=pct_fontsize,
                        color="white" if key != "ambiguous" else "0.10")
        bottoms += vals

    if n_label and totals is not None:
        for xi, t in zip(x, totals):
            ax.text(xi, max(bottoms) + 0.02 * max(bottoms),
                    f"n={t}", ha="center", va="bottom",
                    fontsize=7, color="0.30")
    ax.set_xticks(x)


def main_figure():
    parent_rows = list(csv.DictReader(open(COVERAGE / "per_parent_breakdown.csv")))
    # Skip parents that were not classified (e.g. SEVENLLM was reinstated
    # after the K/A vote; its row is all zeros and would render as an empty bar).
    parent_rows = [r for r in parent_rows if int(r.get("n_samples", 0) or 0) > 0]
    eff = json.loads((CORR / "effective_dimensions.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4),
                                    gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ----- panel (a): per-parent K/A
    labels = [r["parent"] for r in parent_rows]
    K = [int(r["K"]) for r in parent_rows]
    A = [int(r["A"]) for r in parent_rows]
    amb = [int(r["ambiguous"]) for r in parent_rows]
    totals = [int(r["n_samples"]) for r in parent_rows]
    _stacked_with_pct(ax1, labels, K, A, amb, totals=totals, pct_fontsize=7.5)
    ax1.set_xticklabels(labels, rotation=22, ha="right",
                        rotation_mode="anchor", fontsize=8)
    # Tint each parent's tick label using its palette colour.
    for tick, lab in zip(ax1.get_xticklabels(), labels):
        from analysis.lib.plot_style import PARENT_COLOURS
        tick.set_color(PARENT_COLOURS.get(lab, "0.15"))
    ax1.set_ylabel("# samples")
    ax1.set_title("(a) K vs. A composition by parent benchmark", pad=8)
    ax1.legend(loc="upper right", fontsize=7.5, frameon=True,
               bbox_to_anchor=(1.0, 1.0))
    light_grid(ax1, axis="y")
    thin_spines(ax1)

    # ----- panel (b): PCA scree on full 24x10 matrix
    ratio = np.array(eff["explained_variance_ratio"])
    null95 = np.array(eff["null_variance_ratio_p95"])
    n = len(ratio)
    x = np.arange(1, n + 1)
    ax2.bar(x, ratio * 100, color=PALETTE["blue"],
            edgecolor="0.2", linewidth=0.5, label="observed")
    ax2.plot(x, null95 * 100, "x--", color=PALETTE["red"],
             linewidth=1.0, markersize=5.5, label="Horn null (95th pct)")
    ax2.text(1, ratio[0] * 100, f"  PC1 = {ratio[0]*100:.1f}%",
             va="center", ha="left", fontsize=8, color="0.10")
    ax2.set_xticks(x)
    ax2.set_yscale("symlog", linthresh=1.0)
    ax2.set_yticks([0, 1, 5, 10, 50, 100])
    ax2.set_yticklabels(["0", "1", "5", "10", "50", "100"])
    ax2.set_ylim(0, 110)
    ax2.set_xlabel("Component")
    ax2.set_ylabel("% variance explained (symlog)")
    ax2.set_title("(b) PCA scree vs. Horn's parallel-analysis null", pad=8)
    ax2.legend(loc="upper right", fontsize=7.5, frameon=True)
    light_grid(ax2, axis="y")
    thin_spines(ax2)

    fig.savefig(FIGS / "coverage_main.png", dpi=300)
    plt.close(fig)


def main_figure_ka_only():
    """Single-panel K/A composition, sized for an inline wrapfigure (~0.42\\textwidth)."""
    parent_rows = list(csv.DictReader(open(COVERAGE / "per_parent_breakdown.csv")))
    parent_rows = [r for r in parent_rows if int(r.get("n_samples", 0) or 0) > 0]

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    labels = [r["parent"] for r in parent_rows]
    K = [int(r["K"]) for r in parent_rows]
    A = [int(r["A"]) for r in parent_rows]
    amb = [int(r["ambiguous"]) for r in parent_rows]
    totals = [int(r["n_samples"]) for r in parent_rows]
    _stacked_with_pct(ax, labels, K, A, amb, totals=totals, pct_fontsize=6.5)
    ax.set_xticklabels(labels, rotation=35, ha="right",
                       rotation_mode="anchor", fontsize=7.5)
    from analysis.lib.plot_style import PARENT_COLOURS
    for tick, lab in zip(ax.get_xticklabels(), labels):
        tick.set_color(PARENT_COLOURS.get(lab, "0.15"))
    ax.set_ylabel("# samples", fontsize=8)
    ax.legend(loc="upper right", fontsize=6.5, frameon=True,
              handlelength=1.2, handletextpad=0.5, borderpad=0.3)
    light_grid(ax, axis="y")
    thin_spines(ax)
    fig.savefig(FIGS / "coverage_main_ka.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def appendix_figure():
    """Per-sub-task K/A bars, ordered by parent then by definition order."""
    rows = list(csv.DictReader(open(COVERAGE / "per_task_breakdown.csv")))
    by_task = {r["task"]: r for r in rows}

    # Build display order: parents in canonical sequence, sub-tasks in
    # PARENT_GROUPS-defined order, but only those present in the K/A bank.
    ordered_labels = []
    parent_for = {}
    for parent, members in PARENT_GROUPS:
        for t in members:
            if t in by_task:
                ordered_labels.append(t)
                parent_for[t] = parent

    K = [int(by_task[t]["K"]) for t in ordered_labels]
    A = [int(by_task[t]["A"]) for t in ordered_labels]
    amb = [int(by_task[t]["ambiguous"]) for t in ordered_labels]

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    _stacked_with_pct(ax, ordered_labels, K, A, amb, totals=None,
                      bar_width=0.78, pct_fontsize=6, n_label=False)
    ax.set_xticklabels(ordered_labels, rotation=42, ha="right",
                       rotation_mode="anchor", fontsize=7.5)
    # tint by parent
    for tick, lab in zip(ax.get_xticklabels(), ordered_labels):
        tick.set_color(parent_colour(lab, PARENT_GROUPS))
    ax.set_ylabel("# samples")
    ax.set_ylim(0, 110)

    # Parent-group separator lines + labels along the top.
    last_parent = None
    block_starts = []
    for i, t in enumerate(ordered_labels):
        if parent_for[t] != last_parent:
            block_starts.append(i)
            last_parent = parent_for[t]
    block_starts.append(len(ordered_labels))
    for s, e in zip(block_starts[:-1], block_starts[1:]):
        if s > 0:
            ax.axvline(s - 0.5, color="0.65", linewidth=0.6, linestyle=":")
        mid = (s + e - 1) / 2
        parent = parent_for[ordered_labels[s]]
        ax.text(mid, 105, parent, ha="center", va="bottom",
                fontsize=8, color=parent_colour(ordered_labels[s], PARENT_GROUPS),
                fontweight="bold")

    ax.legend(loc="lower right", fontsize=8, frameon=True,
              bbox_to_anchor=(1.0, -0.30), ncol=3)
    light_grid(ax, axis="y")
    thin_spines(ax)

    fig.savefig(FIGS / "coverage_appendix.png", dpi=300)
    plt.close(fig)


def main():
    main_figure()
    main_figure_ka_only()
    appendix_figure()
    print(f"wrote {FIGS}/coverage_main.png + coverage_main_ka.png + coverage_appendix.png")


if __name__ == "__main__":
    main()
