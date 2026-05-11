"""NeurIPS-quality figure: flag-volume sweep curve annotated with FP% at the
agreement-threshold tiers we have search-grounded verifications for.

Design notes:
  - Single panel, single column (≤ 6.7 in wide).
  - Linear y-axis from 0 (avoids the negative-tail ambiguity of log).
  - Open markers for un-audited sweep points, filled red dots for audited tiers.
  - FP% labels float above the audited markers with a 1-line connector.
  - The τ=0.5 audited point — the only partially-covered one — carries an
    explicit denominator so the rate is reported transparently.
  - No in-figure title; LaTeX caption carries the description.

Output: analysis/reports/figures/fp_threshold_combined.png
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from analysis.lib.plot_style import PALETTE, apply_style, light_grid, thin_spines


HERE = Path(__file__).resolve().parent
SWEEP_CSV = HERE / "reports" / "gold_errors" / "threshold_sweep.csv"
VERDICTS_DIR = HERE / "reports" / "verification" / "verdicts" / "search"
FIGS = HERE / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
apply_style()


# ────────────────────────────────────────────────────────────────────────────
# Data loaders
# ────────────────────────────────────────────────────────────────────────────


def threshold_curve():
    rows = list(csv.DictReader(open(SWEEP_CSV)))
    cols = [c for c in rows[0].keys() if c not in {"task", "n_samples"}]
    thr_pairs = [(float(c), c) for c in cols]
    thr_pairs.sort()
    thr = [p[0] for p in thr_pairs]
    totals = [sum(int(r[p[1]]) for r in rows) for p in thr_pairs]
    return thr, totals


def fp_at_threshold():
    """{τ: (n_verified, fp_count, fp_pct)} cumulative across τ ≥ value."""
    samples = []
    for f in VERDICTS_DIR.glob("*.json"):
        d = json.loads(f.read_text())
        samples.append((float(d.get("agreement_fraction", 0.0)),
                        d.get("verdict", "?")))
    out = {}
    for thr in [1.0, 0.9, 0.8333, 0.75, 0.5]:
        verified = [v for af, v in samples if af >= thr - 1e-9]
        n = len(verified)
        if n == 0:
            continue
        fp = sum(1 for v in verified if v == "gold_correct")
        out[thr] = (n, fp, 100.0 * fp / n)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Plot
# ────────────────────────────────────────────────────────────────────────────


def main():
    thr, totals = threshold_curve()
    fp = fp_at_threshold()

    # Single-column NeurIPS-friendly width.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    # ── 1. The trend curve over the full sweep ──────────────────────────────
    ax.plot(thr, totals,
            color="0.55", linewidth=1.0, zorder=2)
    ax.scatter(thr, totals, s=22, facecolors="white",
               edgecolors="0.45", linewidths=0.9, zorder=3,
               label="all sweep thresholds")

    # ── 2. Audited tiers — solid red dots ───────────────────────────────────
    aud_thr = []
    aud_y = []
    aud_pct = []
    aud_labels = []
    for x, (n_ver, n_fp, pct) in fp.items():
        # Some τ values (0.8333) are not in the sweep grid; skip if not present.
        if x not in thr:
            continue
        aud_thr.append(x)
        y = totals[thr.index(x)]
        aud_y.append(y)
        aud_pct.append(pct)
        if abs(x - 0.5) < 1e-6:
            aud_labels.append(f"FP={pct:.0f}%\n({n_ver:,}/{totals[thr.index(x)]:,} audited)")
        else:
            aud_labels.append(f"FP={pct:.0f}%")
    ax.scatter(aud_thr, aud_y, s=48, color=PALETTE["red"],
               edgecolors="white", linewidths=0.8, zorder=5,
               label="audited tiers")

    # ── 3. FP-label boxes above audited dots ────────────────────────────────
    # Compute a vertical offset that scales with the y-range so the labels
    # always sit cleanly above the curve.
    y_max = max(totals)
    label_dy = y_max * 0.08
    for x, y, lab in zip(aud_thr, aud_y, aud_labels):
        ax.annotate(
            lab, xy=(x, y),
            xytext=(x, y + label_dy),
            textcoords="data",
            ha="center", va="bottom",
            fontsize=8, color=PALETTE["red"], fontweight="bold",
            arrowprops=dict(
                arrowstyle="-", color=PALETTE["red"],
                linewidth=0.6, shrinkA=0, shrinkB=4,
                connectionstyle="arc3,rad=0",
            ),
            zorder=6,
        )

    # ── 4. Axes ─────────────────────────────────────────────────────────────
    ax.invert_xaxis()
    ax.set_xlabel(r"Agreement-fraction threshold $\tau$")
    ax.set_ylabel("Total samples flagged")
    ax.set_xlim(1.05, min(thr) - 0.03)
    ax.set_ylim(0, y_max * 1.32)

    # X ticks: only the sweep points we plot.
    ax.set_xticks(thr)
    ax.set_xticklabels([f"{t:.2f}".rstrip("0").rstrip(".") if t > 0.6 else f"{t:.2f}"
                        for t in thr])

    light_grid(ax, axis="y")
    thin_spines(ax)

    ax.legend(loc="upper right", fontsize=7.5,
              handletextpad=0.4, borderpad=0.5)

    fig.tight_layout()
    out = FIGS / "fp_threshold_combined.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
