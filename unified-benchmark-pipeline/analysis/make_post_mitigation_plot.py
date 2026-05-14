"""Post-mitigation generation script.

Outputs:
  /tmp/post_mit_outputs/results_table_cells.tex   — LaTeX cells for filtered tab:main_results
  /tmp/post_mit_outputs/summary_table_rows.tex    — LaTeX rows for the compact summary table
  /tmp/post_mit_outputs/lit_averages.txt          — per-model literature averages, plain text
  /tmp/post_mit_outputs/rank_bump.png             — slope chart figure
  /tmp/post_mit_outputs/numbers.json              — full mapping for downstream sanity-check
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/tmp/post_mit_outputs")
OUT.mkdir(exist_ok=True)

CSV_FILTERED = Path(
    "/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/analysis/reports/label_quality_impact/results_table_filtered.csv"
)
TEX_LIT = Path(
    "/export/home/aberriche/BenchBench/69ca5478b2e0a6c0771198a8/sections/results_table_literature.tex"
)
DELTA_CSV = Path(
    "/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/analysis/reports/label_quality_impact/delta_table.csv"
)

# ----- canonical model order (post-filter rank ascending) ------------------
# This is the column order we'll use in the new results_table.tex AND the bump chart
MODEL_DISPLAY = [
    ("claude-sonnet-4-6-cyberxpert",      "Claude-Sonnet-4.6\\textsuperscript{*}"),
    ("GPT-5.4",                            "GPT-5.4\\textsuperscript{*}"),
    ("Gemma-4-31B-it",                     "Gemma-4-31B"),
    ("Qwen3.6-35B-A3B",                    "Qwen3.6-35B"),
    ("Llama-Primus-Nemotron-70B-Instruct", "Primus-Nemo-70B"),
    ("RedSage-Qwen3-8B-DPO",               "RedSage-8B"),
    ("Llama-3.3-70B-Instruct",             "Llama-3.3-70B"),
    ("GPT-oss-20B",                        "GPT-oss-20B"),
    ("Fanar-2-27B-Instruct",               "Fanar-27B"),
    ("Foundation-Sec-8B-Instruct",         "Found-Sec-8B"),
    ("Llama-Primus-Merged",                "Primus-Merged"),
]
MODEL_KEYS = [k for k, _ in MODEL_DISPLAY]

# ----- 1. Read post-filter CSV ---------------------------------------------
with CSV_FILTERED.open() as f:
    reader = csv.reader(f)
    header = next(reader)
    csv_models = header[1:]
    rows = list(reader)
print(f"Filtered CSV models: {csv_models}")
filtered = {r[0]: {csv_models[i]: float(r[i + 1]) for i in range(len(csv_models))} for r in rows}
sub_tasks = [r[0] for r in rows]
print(f"Filtered CSV sub-tasks: {len(sub_tasks)} → {sub_tasks}")

# Sanity: are all our canonical model keys present?
missing = [m for m in MODEL_KEYS if m not in csv_models]
assert not missing, f"Missing models in CSV: {missing}"

# ----- 2. Per-model post-filter average ------------------------------------
# Match delta_table.csv (which averages over all 24 cells under the unified judge —
# under that judge VSP/RMS are reported as 0-100 strict-verdict scores).
ours_avg = {m: float(np.mean([filtered[t][m] for t in sub_tasks])) for m in MODEL_KEYS}
print("\nOurs (post-filter) per-model average:")
for m in MODEL_KEYS:
    print(f"  {m}: {ours_avg[m]:.2f}")

# ----- 3. Parse literature table ------------------------------------------
# Drop the rows we don't want; keep apples-to-apples Acc/MAD-norm/C+P rows
# Reading as raw text and parsing the data rows
lit_text = TEX_LIT.read_text()
# Each data row pattern:  "Name (with possible \textsuperscript{...}) & val1 & val2 & ... \\"
# Lit column order from header (lines 29-39):
LIT_MODEL_ORDER = [
    "claude-sonnet-4-6-cyberxpert",      # Claude-Sonnet-4.6*
    "GPT-5.4",                            # GPT-5.4*
    "Llama-3.3-70B-Instruct",             # Llama-3.3-70B
    "Qwen3.6-35B-A3B",                    # Qwen3.6-35B
    "Llama-Primus-Nemotron-70B-Instruct", # Primus-Nemo-70B
    "Fanar-2-27B-Instruct",               # Fanar-2-27B
    "RedSage-Qwen3-8B-DPO",               # RedSage-8B-DPO
    "Foundation-Sec-8B-Instruct",         # FoundationSec-8B
    "GPT-oss-20B",                        # GPT-OSS-20B
    "Llama-Primus-Merged",                # Primus-Merged
    "Gemma-4-31B-it",                     # Gemma4-31B
]

# Apples-to-apples row labels we want to KEEP from the literature table.
# Each entry: (display_name, family). We will scan for these literal task labels in the .tex.
LIT_KEEP_LABELS = [
    # CTI-Bench
    ("MCQ", "CTI-Bench"),
    ("RCM", "CTI-Bench"),
    ("ATE", "CTI-Bench"),
    ("TAA", "CTI-Bench"),  # C+P metric — comparable in spirit
    # AthenaBench
    ("CKT", "AthenaBench"),
    ("TAA", "AthenaBench"),
    ("ATE", "AthenaBench"),
    ("RCM", "AthenaBench"),
    # SECURE
    ("MAET", "SECURE"),
    ("CWET", "SECURE"),
    ("KCV", "SECURE"),
    # General
    ("SecEval$_\\text{5tok}$", "General"),
    ("CyberMetric$_\\text{samp}$", "General"),
    ("MMLU-CS$_\\text{gen}$", "General"),
    ("SecBench", "General"),
    # RedSage-MCQ (ours, generative) — apples-to-apples to our judge
    ("CLI",    "RedSage-MCQ (ours, generative)"),
    ("FW",     "RedSage-MCQ (ours, generative)"),
    ("GEN",    "RedSage-MCQ (ours, generative)"),
    ("Kali",   "RedSage-MCQ (ours, generative)"),
    ("Skills", "RedSage-MCQ (ours, generative)"),
]

# Also rows we keep but exclude from averaging (different metric scale)
# Already-in-lit-table rows: VSP MAD↓, RMS F1, AthenaBench VSP MAD-norm, CTI-Bench VSP MAD↓
# We ONLY average over Acc-percentage cells (and MAD-norm which IS 0-100 acc-scale)
EXCLUDE_FROM_LIT_AVG_KEYWORDS = {"VSP", "RMS"}  # raw MAD or F1 — not avg-able

# We'll parse the lit table line-by-line. For each row: extract the leading task label
# and the 11 numeric cells.

def parse_cell(s: str) -> float | None:
    """Parse a single cell. Strip \\textbf{...}, \\ensuremath{...}, \\textsuperscript{...}.
    Return None if cell is '--' or empty.
    """
    s = s.strip()
    if not s or s == "--":
        return None
    # remove \textbf{...} → keep inner
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    # remove \ensuremath{...} marks (footnotes like ^\dagger, ^\ddagger)
    s = re.sub(r"\\ensuremath\{[^}]*\}", "", s)
    s = re.sub(r"\\textsuperscript\{[^}]*\}", "", s)
    s = s.strip()
    # try float
    try:
        return float(s)
    except ValueError:
        return None


# Parse data rows. Some rows span multiple physical lines in source; we accumulate
# until we hit a line ending in '\\' that, joined with prior lines, contains 11 '&'.
data_rows: list[tuple[str, str, list[float | None]]] = []
current_section = None

# Pre-split on rows ending in literal '\\\\'
lines = lit_text.splitlines()
i = 0
while i < len(lines):
    s = lines[i].strip()
    if not s or s.startswith("%"):
        i += 1
        continue
    sec = re.match(r"\\multicolumn\{12\}\{l\}\{\\textit\{([^}]*)\}\}\\\\", s)
    if sec:
        current_section = sec.group(1)
        i += 1
        continue
    if "\\midrule" in s or "\\toprule" in s or "\\bottomrule" in s or s.startswith("\\cmidrule"):
        i += 1
        continue
    # Accumulate a logical row across lines until trailing '\\'.
    # The first physical line may not contain '&' (label-only line).
    chunks = [s]
    while not chunks[-1].rstrip().endswith("\\\\") and i + 1 < len(lines):
        i += 1
        chunks.append(lines[i].strip())
    body = " ".join(chunks)
    if not body.rstrip().endswith("\\\\"):
        i += 1
        continue
    body = body.rstrip()[:-2].strip()
    parts = [p.strip() for p in re.split(r"(?<!\\)&", body)]
    if len(parts) != 12:
        i += 1
        continue
    label = parts[0].strip()
    label_clean = re.sub(r"\\textsuperscript\{[^}]*\}", "", label).strip()
    if label_clean.lower().startswith("\\textit{average}") or label_clean.lower().startswith("\\textit{avg.}"):
        i += 1
        continue
    vals = [parse_cell(p) for p in parts[1:]]
    data_rows.append((label_clean, current_section or "?", vals))
    i += 1

print(f"\nParsed {len(data_rows)} literature data rows across sections.")
for lab, fam, _ in data_rows:
    print(f"  {fam}: {lab}")

# Filter to apples-to-apples
KEEP_PAIRS = set((lab, fam) for lab, fam in LIT_KEEP_LABELS)
kept_rows = [(lab, fam, vals) for lab, fam, vals in data_rows if (lab, fam) in KEEP_PAIRS]
print(f"\nKept apples-to-apples rows: {len(kept_rows)}")
for lab, fam, _ in kept_rows:
    print(f"  {fam}: {lab}")

# ----- 4. Compute literature per-model average ----------------------------
# Per-model: average over kept rows EXCLUDING raw-MAD and F1 rows.
# The kept rows are all Acc/MAD-norm/C+P, so all on a 0-100 scale comparable to Acc.
lit_avg = {}
for i, m in enumerate(LIT_MODEL_ORDER):
    vals = [r[2][i] for r in kept_rows if r[2][i] is not None]
    lit_avg[m] = float(np.mean(vals)) if vals else float("nan")
    print(f"Lit avg [{m}]: {lit_avg[m]:.2f} (n={len(vals)} cells)")

# ----- 5. Compute literature ranks + delta-rank ---------------------------
lit_rank = {m: r + 1 for r, m in enumerate(sorted(MODEL_KEYS, key=lambda mm: -lit_avg[mm]))}
ours_rank = {m: r + 1 for r, m in enumerate(sorted(MODEL_KEYS, key=lambda mm: -ours_avg[mm]))}

print("\nRank summary:")
print(f"{'Model':30} {'Lit Avg':>8} {'Lit Rk':>7} {'Ours Avg':>8} {'Ours Rk':>7} {'Δ pp':>7} {'Δ Rk':>6}")
for m in MODEL_KEYS:
    drk = lit_rank[m] - ours_rank[m]
    print(f"{m:30} {lit_avg[m]:>8.2f} {lit_rank[m]:>7d} {ours_avg[m]:>8.2f} {ours_rank[m]:>7d} {ours_avg[m]-lit_avg[m]:+7.2f} {drk:+6d}")

# ----- 6. Bump chart -------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.2))
n = len(MODEL_KEYS)
x_lit = 0
x_ours = 1
ax.set_xlim(-0.7, 1.7)
ax.set_ylim(0.0, n + 1.2)
ax.invert_yaxis()  # rank 1 at top

# Sort by ours_rank for consistent placement
disp_lit = sorted(MODEL_KEYS, key=lambda m: lit_rank[m])
disp_ours = sorted(MODEL_KEYS, key=lambda m: ours_rank[m])

# Draw model labels at lit and ours columns
LABEL = dict(MODEL_DISPLAY)
for m in MODEL_KEYS:
    yl = lit_rank[m]
    yr = ours_rank[m]
    delta = lit_rank[m] - ours_rank[m]
    if delta >= 2:
        color = "#1a8c3a"  # green up
        lw = 1.6
    elif delta <= -2:
        color = "#c0392b"  # red down
        lw = 1.6
    else:
        color = "#888888"
        lw = 0.9
    ax.plot([x_lit, x_ours], [yl, yr], color=color, linewidth=lw, alpha=0.85, zorder=2)

# Markers
for m in MODEL_KEYS:
    ax.plot(x_lit, lit_rank[m], "o", color="0.2", markersize=3, zorder=3)
    ax.plot(x_ours, ours_rank[m], "o", color="0.2", markersize=3, zorder=3)

# Labels: lit on left, ours on right
disp_label_lit = {m: LABEL[m].replace("\\textsuperscript{*}", "*") for m in MODEL_KEYS}
for m in MODEL_KEYS:
    ax.text(x_lit - 0.05, lit_rank[m], f"{lit_rank[m]}. {disp_label_lit[m]}",
            ha="right", va="center", fontsize=8)
for m in MODEL_KEYS:
    delta = lit_rank[m] - ours_rank[m]
    suffix = ""
    if delta != 0:
        suffix = f"  ({'+' if delta > 0 else ''}{delta})"
    ax.text(x_ours + 0.05, ours_rank[m], f"{ours_rank[m]}. {disp_label_lit[m]}{suffix}",
            ha="left", va="center", fontsize=8)

ax.set_xticks([x_lit, x_ours])
ax.set_xticklabels(["Literature", "Unified harness\n(post-mitigation)"], fontsize=9)
ax.set_yticks([])
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="x", length=0)

# Legend hint at the very top, title moved to bottom for clarity
ax.text(0.5, 0.2,
        "Green = climbed $\\geq$2 ranks   |   Red = dropped $\\geq$2 ranks   |   Grey = $|\\Delta| \\leq 1$",
        ha="center", va="top", fontsize=7, color="0.3", transform=ax.transData)

fig.tight_layout()
fig.savefig(OUT / "rank_bump.png", dpi=300, bbox_inches="tight")
print(f"Saved bump chart: {OUT / 'rank_bump.png'}")

# ----- 6b. Paired bar chart: Lit vs Ours per model -------------------------
# Sorted by Ours avg descending so the leaderboard reads top-to-bottom.
order = sorted(MODEL_KEYS, key=lambda mm: -ours_avg[mm])
n = len(order)
fig2, ax2 = plt.subplots(figsize=(5.6, 4.0))
y = np.arange(n)
bar_h = 0.38
lit_vals  = [lit_avg[m]  for m in order]
ours_vals = [ours_avg[m] for m in order]

ax2.barh(y - bar_h / 2, lit_vals,  bar_h, color="#bbbbbb", edgecolor="0.3", linewidth=0.5, label="Literature")
ax2.barh(y + bar_h / 2, ours_vals, bar_h, color="#3a7bd5", edgecolor="0.2", linewidth=0.5, label="Unified harness (post-mit.)")

# Δpp annotations to the right of the longer bar of each pair
for i, m in enumerate(order):
    longer = max(lit_avg[m], ours_avg[m])
    delta = ours_avg[m] - lit_avg[m]
    drk   = lit_rank[m] - ours_rank[m]
    sign  = "+" if delta >= 0 else ""
    drk_s = f"{drk:+d}" if drk != 0 else "0"
    color = "#1a8c3a" if drk >= 2 else ("#c0392b" if drk <= -2 else "0.3")
    ax2.text(longer + 1.5, i, f"{sign}{delta:.1f} pp  ({drk_s})",
             va="center", fontsize=7.5, color=color)

ax2.set_yticks(y)
ax2.set_yticklabels([LABEL[m].replace("\\textsuperscript{*}", "*") for m in order], fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel("Average accuracy (%)", fontsize=9)
ax2.set_xlim(0, 110)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
           fontsize=7.5, frameon=False)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
ax2.tick_params(axis="x", labelsize=8)
ax2.grid(axis="x", which="major", color="0.85", linewidth=0.5, zorder=0)

fig2.tight_layout()
fig2.savefig(OUT / "post_mit_barplot.png", dpi=300, bbox_inches="tight")
print(f"Saved bar chart: {OUT / 'post_mit_barplot.png'}")

# ----- 7. Compact summary table LaTeX -------------------------------------
summary_lines = []
for m in MODEL_KEYS:
    name = LABEL[m]
    lavg = lit_avg[m]
    oavg = ours_avg[m]
    dpp = oavg - lavg
    drk = lit_rank[m] - ours_rank[m]
    drk_s = f"{drk:+d}" if drk != 0 else "0"
    summary_lines.append(
        f"{name:<35} & {lavg:5.1f} & {oavg:5.1f} & {dpp:+5.1f} & {lit_rank[m]} $\\to$ {ours_rank[m]} ({drk_s}) \\\\"
    )
(OUT / "summary_table_rows.tex").write_text("\n".join(summary_lines) + "\n")
print("\nSummary table rows:")
print((OUT / "summary_table_rows.tex").read_text())

# ----- 8. New results_table.tex cells (filtered) --------------------------
# Reuse existing structure; just emit rows in same order as current table.
# Bottom "Average" row uses ours_avg (already computed).

# All cells under the unified GPT-5.4 judge are reported as 0-100 strict-verdict
# scores. We use a uniform metric superscript (Acc) for all rows; the bottom
# Average row is the per-model mean across all Acc cells.
ROW_ORDER = [
    ("ate",        "CTI-Bench",   "ATE"),
    ("rcm",        "CTI-Bench",   "RCM"),
    ("vsp",        "CTI-Bench",   "VSP"),
    ("mcq",        "CTI-Bench",   "MCQ"),
    ("cti_taa",    "CTI-Bench",   "TAA"),
    ("ckt",        "AthenaBench", "CKT"),
    ("rms",        "AthenaBench", "RMS"),
    ("taa",        "AthenaBench", "TAA"),
    ("athena_ate", "AthenaBench", "ATE"),
    ("athena_rcm", "AthenaBench", "RCM"),
    ("athena_vsp", "AthenaBench", "VSP"),
    ("secure_maet","SECURE",      "MAET"),
    ("secure_cwet","SECURE",      "CWET"),
    ("secure_kcv", "SECURE",      "KCV"),
    ("redsage_frameworks", "RedSage-MCQ", "FW"),
    ("redsage_generals",   "RedSage-MCQ", "GEN"),
    ("redsage_skills",     "RedSage-MCQ", "Skills"),
    ("redsage_cli",        "RedSage-MCQ", "CLI"),
    ("redsage_kali",       "RedSage-MCQ", "Kali"),
    ("cybermetric",  "CyberMetric", "CyberMetric"),
    ("mmlu_cs",      "MMLU-CS",     "MMLU-CS"),
    ("secbench",     "SecBench",    "SecBench"),
    ("seceval",      "SecEval",     "SecEval"),
    ("sevenllm",     "SEvenLLM",    "SEvenLLM"),
]

def fmt_cell(value: float, is_best: bool, fmt: str = "{:.1f}") -> str:
    body = fmt.format(value)
    return f"\\textbf{{{body}}}" if is_best else body

# Build the table body
tex_lines = []
last_family = None
for csv_task, fam, disp_label in ROW_ORDER:
    if fam != last_family:
        tex_lines.append("\\midrule")
        tex_lines.append(f"\\multicolumn{{12}}{{l}}{{\\textit{{{fam}}}}}\\\\")
        last_family = fam
    vals = [filtered[csv_task][m] for m in MODEL_KEYS]
    best = max(vals)
    cells = [fmt_cell(v, v == best) for v in vals]
    tex_lines.append(f"{disp_label} & " + " & ".join(cells) + " \\\\")

# Bottom average row over all 24 cells (per-model mean of Acc scores)
tex_lines.append("\\midrule")
avg_all = {m: float(np.mean([filtered[t][m] for t in sub_tasks])) for m in MODEL_KEYS}
avg_vals = [avg_all[m] for m in MODEL_KEYS]
best = max(avg_vals)
cells = [fmt_cell(v, v == best) for v in avg_vals]
tex_lines.append("\\textit{Average} & " + " & ".join(cells) + " \\\\")

(OUT / "results_table_cells.tex").write_text("\n".join(tex_lines) + "\n")
print("\nGenerated results_table cells (head):")
print("\n".join(tex_lines[:6]))

# ----- 9. Dump everything to JSON for sanity check -------------------------
out = {
    "ours_avg": ours_avg,
    "ours_rank": ours_rank,
    "lit_avg": lit_avg,
    "lit_rank": lit_rank,
    "kept_lit_rows": [(lab, fam) for lab, fam, _ in kept_rows],
    "n_apples_rows": len(kept_rows),
}
(OUT / "numbers.json").write_text(json.dumps(out, indent=2))
print(f"\nWrote numbers.json with {len(kept_rows)} apples-to-apples rows")
