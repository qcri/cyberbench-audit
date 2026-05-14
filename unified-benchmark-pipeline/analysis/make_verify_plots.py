"""Plots for the label-quality verification section. NeurIPS-quality output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.lib.plot_style import (
    PALETTE,
    apply_style,
    light_grid,
    thin_spines,
)


HERE = Path(__file__).resolve().parent
VERIF = HERE / "reports" / "verification"
FIGS = HERE / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

apply_style()


VERDICTS = ["gold_correct", "majority_correct", "both_wrong", "uncertain"]
VERDICT_LABEL = {
    "gold_correct":     "gold-correct (FP)",
    "majority_correct": "gold-mislabel (TP)",
    "both_wrong":       "both wrong",
    "uncertain":        "uncertain",
}
COLOURS = {
    "gold_correct":     PALETTE["green"],
    "majority_correct": PALETTE["red"],
    "both_wrong":       PALETTE["orange"],
    "uncertain":        PALETTE["grey"],
}


def load_rows(agent: str):
    path = VERIF / f"per_threshold_{agent}.csv"
    if not path.exists():
        return []
    return list(csv.DictReader(open(path)))


def plot_fp_vs_threshold():
    s = load_rows("search")
    d = load_rows("direct")
    if not s and not d:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    series = [
        ("search-grounded verifier", s, PALETTE["blue"], "o"),
        ("direct verifier",          d, PALETTE["red"],  "s"),
    ]
    for label, rows, colour, marker in series:
        if not rows:
            continue
        thr = [float(r["threshold"]) for r in rows]
        flagged = [int(r["n_flagged"]) for r in rows]
        verified = [int(r["n_verified"]) for r in rows]
        fp_rate = [int(r["gold_correct"]) / max(int(r["n_verified"]), 1) * 100
                   for r in rows]
        ax.plot(thr, fp_rate, marker=marker, color=colour, label=label,
                linewidth=1.4, markersize=6, markeredgecolor="white",
                markeredgewidth=0.8)
        for x, y, n_v, n_f in zip(thr, fp_rate, verified, flagged):
            offset = (4, 6) if colour == PALETTE["blue"] else (4, -12)
            ax.annotate(f"n={n_f}", (x, y),
                        textcoords="offset points", xytext=offset,
                        fontsize=7, color=colour)
    # Highlight productive triage zone tau in [0.75, 0.90]
    ax.axvspan(0.75, 0.90, color=PALETTE["green"], alpha=0.10, zorder=0)
    ax.text(0.825, 78, "productive\ntriage zone",
            ha="center", va="top", fontsize=7.5, color=PALETTE["green"],
            alpha=0.95)
    ax.axhline(50, color="0.6", linewidth=0.5, linestyle=":")
    ax.set_xticks([1.00, 0.90, 0.833, 0.75])
    ax.set_xticklabels(["1.000", "0.900", "0.833", "0.750"])
    ax.invert_xaxis()
    ax.set_xlabel(r"agreement-fraction threshold $\tau$ (high $\to$ low)")
    ax.set_ylabel("false-positive rate (%) of flagging procedure")
    ax.set_ylim(35, 85)
    light_grid(ax, axis="y")
    thin_spines(ax)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.savefig(FIGS / "fp_vs_threshold.png", dpi=300)
    plt.close(fig)


def plot_verdict_breakdown():
    rows_by_agent = {a: load_rows(a) for a in ["search", "direct"]}
    rows_by_agent = {k: v for k, v in rows_by_agent.items() if v}
    if not rows_by_agent:
        return

    n_panels = len(rows_by_agent)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.4 * n_panels, 3.6),
                              sharey=True)
    if n_panels == 1:
        axes = [axes]
    for ax, (agent, rows) in zip(axes, rows_by_agent.items()):
        thr_labels = [r["threshold"] for r in rows]
        x = np.arange(len(thr_labels))
        bottoms = np.zeros(len(thr_labels))
        any_pending = sum(int(r["pending"]) for r in rows) > 0
        for v in VERDICTS:
            heights = np.array([int(r[v]) for r in rows], dtype=float)
            if heights.sum() == 0:
                continue
            ax.bar(x, heights, bottom=bottoms, label=VERDICT_LABEL[v],
                   color=COLOURS[v], edgecolor="white", linewidth=0.4)
            for i, (h, b) in enumerate(zip(heights, bottoms)):
                if h >= 8:
                    ax.text(x[i], b + h / 2, str(int(h)),
                            ha="center", va="center", fontsize=6.8,
                            color="white" if v != "uncertain" else "0.10")
            bottoms += heights
        if any_pending:
            pend = np.array([int(r["pending"]) for r in rows], dtype=float)
            ax.bar(x, pend, bottom=bottoms, label="pending", color="white",
                   edgecolor="0.4", linewidth=0.6, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(thr_labels)
        ax.invert_xaxis()
        ax.set_xlabel(r"agreement-fraction threshold $\tau$")
        title = "search-grounded verifier" if agent == "search" else "direct verifier"
        ax.set_title(title, fontsize=9.5)
        light_grid(ax, axis="y")
        thin_spines(ax)
    axes[0].set_ylabel("# flags")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=8, frameon=False)
    fig.savefig(FIGS / "verdict_breakdown.png", dpi=300)
    plt.close(fig)


def plot_agent_agreement():
    path = VERIF / "agreement.csv"
    if not path.exists():
        return
    rows = list(csv.reader(open(path)))
    header = rows[0][1:-1]
    M = np.array([[int(c) for c in r[1:-1]] for r in rows[1:len(VERDICTS)+1]])
    if M.size == 0:
        return

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(M, cmap="Blues", aspect="equal")
    ax.set_xticks(range(len(header)))
    ax.set_xticklabels(header, rotation=22, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(VERDICTS)))
    ax.set_yticklabels([f"search = {VERDICT_LABEL[v].split(' (')[0]}" for v in VERDICTS])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "0.10",
                    fontsize=8)
    ax.set_xlabel("direct verifier verdict")
    ax.set_ylabel("search-grounded verifier verdict")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=7.5)
    fig.savefig(FIGS / "agent_agreement.png", dpi=300)
    plt.close(fig)


def main():
    plot_fp_vs_threshold()
    plot_verdict_breakdown()
    plot_agent_agreement()
    print(f"figures written to {FIGS}/")


if __name__ == "__main__":
    main()
