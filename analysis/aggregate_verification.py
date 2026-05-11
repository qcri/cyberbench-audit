"""Aggregate per-sample verdicts into per-threshold FP / TP counts.

Reads:
    analysis/reports/verification/verdicts/<agent>/<task>_<idx>.json

Writes:
    analysis/reports/verification/per_threshold_<agent>.csv
    analysis/reports/verification/agreement.csv          (search vs direct)
    analysis/reports/verification/disagreement_samples.jsonl
    analysis/reports/verification/summary.md             (concise narrative)
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERIF_DIR = HERE / "reports" / "verification"
BANK_PATH = VERIF_DIR / "flagged_bank.jsonl"

THRESHOLDS = [1.00, 0.90, 5/6, 0.75]
THRESH_LABEL = {1.00: "1.000", 0.90: "0.900", 5/6: "0.833", 0.75: "0.750"}
VERDICTS = ["gold_correct", "majority_correct", "both_wrong", "uncertain"]


def load_verdicts(agent: str) -> dict:
    by_id = {}
    d = VERIF_DIR / "verdicts" / agent
    if not d.exists():
        return by_id
    for f in d.glob("*.json"):
        try:
            v = json.loads(f.read_text())
            by_id[(v["task"], str(v["index"]))] = v
        except Exception:
            continue
    return by_id


def aggregate_one_agent(agent: str, bank):
    verdicts = load_verdicts(agent)
    out_rows = []
    for thr in THRESHOLDS:
        eligible = [r for r in bank if r["agreement_fraction"] >= thr - 1e-9]
        n_total = len(eligible)
        counts = Counter()
        n_verified = 0
        for r in eligible:
            v = verdicts.get((r["task"], str(r["index"])))
            if v is None:
                counts["pending"] += 1
            else:
                counts[v["verdict"]] += 1
                n_verified += 1
        row = {
            "threshold": THRESH_LABEL[thr],
            "n_flagged": n_total,
            "n_verified": n_verified,
            **{v: counts.get(v, 0) for v in VERDICTS},
            "pending": counts.get("pending", 0),
        }
        # FP rate over verified samples (gold_correct = flag was a false positive)
        row["fp_rate_verified"] = (
            round(counts.get("gold_correct", 0) / n_verified, 4)
            if n_verified else 0.0
        )
        out_rows.append(row)

    out = VERIF_DIR / f"per_threshold_{agent}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def agreement_matrix(bank):
    s = load_verdicts("search")
    d = load_verdicts("direct")
    common_keys = sorted(set(s.keys()) & set(d.keys()))
    matrix = defaultdict(int)
    for k in common_keys:
        matrix[(s[k]["verdict"], d[k]["verdict"])] += 1

    out_path = VERIF_DIR / "agreement.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", *VERDICTS, "row_total"])
        for sv in VERDICTS:
            row = [f"search={sv}"]
            row_tot = 0
            for dv in VERDICTS:
                n = matrix.get((sv, dv), 0)
                row.append(n)
                row_tot += n
            row.append(row_tot)
            w.writerow(row)

    disagree_path = VERIF_DIR / "disagreement_samples.jsonl"
    bank_idx = {(r["task"], str(r["index"])): r for r in bank}
    with open(disagree_path, "w") as f:
        for k in common_keys:
            if s[k]["verdict"] != d[k]["verdict"]:
                rec = {
                    "task": k[0],
                    "index": k[1],
                    "gold": s[k]["gold"],
                    "majority_prediction": s[k]["majority_prediction"],
                    "agreement_fraction": s[k]["agreement_fraction"],
                    "search_verdict": s[k]["verdict"],
                    "search_confidence": s[k].get("confidence"),
                    "search_justification": s[k].get("justification"),
                    "search_citations": s[k].get("citations", []),
                    "direct_verdict": d[k]["verdict"],
                    "direct_confidence": d[k].get("confidence"),
                    "direct_justification": d[k].get("justification"),
                    "question": bank_idx.get(k, {}).get("question", "")[:600],
                }
                json.dump(rec, f); f.write("\n")
    return matrix


def write_summary(bank, search_rows, direct_rows, matrix):
    lines = ["# Verification summary", ""]
    lines.append("Two GPT-5.4 verifiers were run over the flagged-sample bank.")
    lines.append("`search` is grounded with the Azure web-search tool restricted to a tier-1/2 cybersec whitelist; `direct` uses model knowledge alone.")
    lines.append("")
    lines.append("## Per-threshold breakdown")
    lines.append("")
    for label, rows in [("search", search_rows), ("direct", direct_rows)]:
        lines.append(f"### Agent: {label}")
        lines.append("")
        lines.append("| threshold | n_flagged | n_verified | gold_correct (FP) | majority_correct (TP) | both_wrong | uncertain | pending | FP-rate (verified) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r['threshold']} | {r['n_flagged']} | {r['n_verified']} | "
                f"{r['gold_correct']} | {r['majority_correct']} | {r['both_wrong']} | "
                f"{r['uncertain']} | {r['pending']} | {r['fp_rate_verified']} |"
            )
        lines.append("")

    if matrix:
        common = sum(matrix.values())
        agree = sum(matrix.get((v, v), 0) for v in VERDICTS)
        lines.append("## Search vs direct agreement")
        lines.append("")
        lines.append(f"common verified samples: **{common}**, identical verdicts: **{agree}** ({100*agree/common:.1f}%)")
        lines.append("")
        lines.append("| | " + " | ".join(VERDICTS) + " |")
        lines.append("|" + "|".join(["---"] * (len(VERDICTS) + 1)) + "|")
        for sv in VERDICTS:
            row = [f"search={sv}"]
            for dv in VERDICTS:
                row.append(str(matrix.get((sv, dv), 0)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    (VERIF_DIR / "summary.md").write_text("\n".join(lines))


def main():
    bank = [json.loads(l) for l in BANK_PATH.open() if l.strip()]
    search_rows = aggregate_one_agent("search", bank)
    direct_rows = aggregate_one_agent("direct", bank)
    matrix = agreement_matrix(bank)
    write_summary(bank, search_rows, direct_rows, matrix)
    print(f"wrote {VERIF_DIR}/per_threshold_*.csv + summary.md + agreement.csv")


if __name__ == "__main__":
    main()
