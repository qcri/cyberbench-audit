"""Aggregate per-(model, sample) K/A verdicts into majority votes + breakdowns.

Inputs: analysis/reports/coverage/verdicts/<model>/<task>_<idx>.json (4 models)
Outputs:
  per_sample_verdicts.csv
  per_task_breakdown.csv      (counts per sub-task: K, A, ambiguous, total)
  per_parent_breakdown.csv    (sample-weighted aggregation by parent)
  agreement.csv               (Cohen's κ pairwise + Fleiss's κ)
  summary.md                  (human-readable narrative)
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from analysis.lib.loaders import PARENT_GROUPS, TASK_ORDER


HERE = Path(__file__).resolve().parent
COVERAGE = HERE / "reports" / "coverage"
VERDICTS = COVERAGE / "verdicts"

MODELS = ["GPT-5.4", "Qwen3.6-35B-A3B", "RedSage-Qwen3-8B-DPO", "Llama-3.3-70B-Instruct"]
CLASSES = ["K", "A"]


def load_verdicts() -> Dict[str, Dict]:
    """Return: {(task, idx) -> {model_label: class}}."""
    out: Dict[str, Dict[str, str]] = {}
    for model in MODELS:
        d = VERDICTS / model
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            try:
                v = json.loads(f.read_text())
            except Exception:
                continue
            cls = v.get("class")
            if cls not in CLASSES:
                continue
            key = (v["task"], str(v["index"]))
            out.setdefault(key, {})[model] = cls
    return out


def majority_label(verdicts: Dict[str, str]) -> str:
    """Return 'K', 'A', or 'ambiguous'."""
    if not verdicts:
        return "abstain"
    c = Counter(verdicts.values())
    top, n = c.most_common(1)[0]
    if n >= 3:  # 3 or 4 of 4 models agree
        return top
    if n == 2 and len(c) == 2:
        return "ambiguous"  # 2-2 split
    return top  # 2 votes for one class with 0 or 1 abstain on the other → take the lead


def parent_of(task: str) -> str:
    for parent, members in PARENT_GROUPS:
        if task in members:
            return parent
    return "OTHER"


# ---------------- κ helpers ----------------

def cohens_kappa(a: List[str], b: List[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    pa = sum(1 for x, y in zip(a, b) if x == y) / n
    counts_a = Counter(a)
    counts_b = Counter(b)
    pe = sum((counts_a.get(c, 0) / n) * (counts_b.get(c, 0) / n) for c in CLASSES)
    if pe >= 1:
        return float("nan")
    return (pa - pe) / (1 - pe)


def fleiss_kappa(rater_matrix: List[List[str]]) -> float:
    """rater_matrix: list of lists, where rater_matrix[i] is the verdict per item by rater i."""
    if not rater_matrix:
        return float("nan")
    n_items = len(rater_matrix[0])
    if n_items == 0:
        return float("nan")
    R = len(rater_matrix)
    # Build (n_items × |classes|) matrix of category counts
    p_j = {c: 0.0 for c in CLASSES}
    P_i = []
    valid_items = 0
    for i in range(n_items):
        votes = [rater_matrix[r][i] for r in range(R) if rater_matrix[r][i] in CLASSES]
        if len(votes) < 2:
            continue
        n = len(votes)
        c = Counter(votes)
        for cl in CLASSES:
            p_j[cl] += c.get(cl, 0) / n
        P_i.append(
            (sum(c.get(cl, 0) ** 2 for cl in CLASSES) - n) / max(1, n * (n - 1))
        )
        valid_items += 1
    if valid_items == 0:
        return float("nan")
    p_j = {cl: p_j[cl] / valid_items for cl in CLASSES}
    P_bar = sum(P_i) / valid_items
    P_e = sum(p_j[cl] ** 2 for cl in CLASSES)
    if P_e >= 1:
        return float("nan")
    return (P_bar - P_e) / (1 - P_e)


# ---------------- main ----------------

def main():
    verdicts = load_verdicts()
    if not verdicts:
        raise SystemExit("No verdicts found. Run classify_ka first.")

    keys = sorted(verdicts.keys(), key=lambda k: (k[0], int(k[1]) if k[1].isdigit() else 0))

    # per-sample CSV
    rows = []
    for task, idx in keys:
        v = verdicts[(task, idx)]
        rows.append({
            "task": task,
            "index": idx,
            **{m: v.get(m, "") for m in MODELS},
            "majority": majority_label(v),
            "n_votes": len(v),
        })
    with open(COVERAGE / "per_sample_verdicts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # per-task breakdown
    by_task: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_task[r["task"]][r["majority"]] += 1
    task_rows = []
    for t in TASK_ORDER:
        c = by_task.get(t, Counter())
        n = sum(c.values())
        task_rows.append({
            "task": t,
            "parent": parent_of(t),
            "n_samples": n,
            "K": c.get("K", 0),
            "A": c.get("A", 0),
            "ambiguous": c.get("ambiguous", 0),
            "abstain": c.get("abstain", 0),
            "K_pct": round(100 * c.get("K", 0) / n, 1) if n else 0,
            "A_pct": round(100 * c.get("A", 0) / n, 1) if n else 0,
        })
    with open(COVERAGE / "per_task_breakdown.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()))
        w.writeheader(); w.writerows(task_rows)

    # per-parent breakdown (sample-weighted average)
    by_parent: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_parent[parent_of(r["task"])][r["majority"]] += 1
    parent_rows = []
    for parent, _ in PARENT_GROUPS:
        c = by_parent.get(parent, Counter())
        n = sum(c.values())
        parent_rows.append({
            "parent": parent,
            "n_samples": n,
            "K": c.get("K", 0),
            "A": c.get("A", 0),
            "ambiguous": c.get("ambiguous", 0),
            "K_pct": round(100 * c.get("K", 0) / n, 1) if n else 0,
            "A_pct": round(100 * c.get("A", 0) / n, 1) if n else 0,
        })
    with open(COVERAGE / "per_parent_breakdown.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(parent_rows[0].keys()))
        w.writeheader(); w.writerows(parent_rows)

    # agreement: Cohen's κ pairwise + Fleiss's κ
    matrix = []
    for m in MODELS:
        col = [verdicts[k].get(m, "ABSTAIN") for k in keys]
        matrix.append(col)

    agreement_rows = [["pair", "n", "kappa"]]
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            a, b = matrix[i], matrix[j]
            both = [(x, y) for x, y in zip(a, b) if x in CLASSES and y in CLASSES]
            n = len(both)
            if n == 0:
                k = float("nan")
            else:
                k = cohens_kappa([x for x, _ in both], [y for _, y in both])
            agreement_rows.append([f"{MODELS[i]} ↔ {MODELS[j]}", n, round(k, 4)])
    fl = fleiss_kappa(matrix)
    agreement_rows.append(["Fleiss (all 4)", len(keys), round(fl, 4)])
    with open(COVERAGE / "agreement.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerows(agreement_rows)

    # summary markdown
    n_items = len(keys)
    n_per_model = {m: sum(1 for k in keys if m in verdicts[k]) for m in MODELS}
    summary = ["# K-vs-A coverage classification — summary", ""]
    summary.append(f"- bank: **{n_items}** samples (~100 per sub-task; ate has 55, taa 100, mmlu_cs 100).")
    summary.append("- models (top-4 by overall accuracy from §3.2.1; Llama-3.3-70B-Instruct substituted for Gemma-4-31B-it which is not locally cached):")
    for m in MODELS:
        summary.append(f"  - {m}: {n_per_model[m]} verdicts cached")
    summary.append("")
    summary.append("## Inter-model agreement")
    summary.append("")
    summary.append("| pair | n | Cohen's κ |")
    summary.append("|---|---|---|")
    for r in agreement_rows[1:-1]:
        summary.append(f"| {r[0]} | {r[1]} | {r[2]} |")
    summary.append("")
    summary.append(f"**Fleiss's κ (4 raters, n = {n_items}): {fl:.3f}**")
    summary.append("")
    summary.append("## Per-parent K/A composition")
    summary.append("")
    summary.append("| parent | n | K | A | ambiguous | K % | A % |")
    summary.append("|---|---|---|---|---|---|---|")
    for r in parent_rows:
        summary.append(f"| {r['parent']} | {r['n_samples']} | {r['K']} | {r['A']} | {r['ambiguous']} | {r['K_pct']} | {r['A_pct']} |")
    (COVERAGE / "summary.md").write_text("\n".join(summary))

    print(f"wrote {COVERAGE}/per_sample_verdicts.csv, per_task_breakdown.csv, "
          f"per_parent_breakdown.csv, agreement.csv, summary.md")
    print(f"  Fleiss's κ = {fl:.3f}")


if __name__ == "__main__":
    main()
