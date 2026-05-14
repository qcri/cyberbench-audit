"""Pick one or two representative mislabel examples per benchmark and emit a
LaTeX block of `tcolorbox` examples for the appendix.

Selection rule: from `verdicts/search/`, take verdicts where verdict ==
`majority_correct` (the search agent says the gold is wrong), prefer high
agreement-fraction items (more model consensus -> better example), prefer
items with at least one citation. Group by task and emit at most TWO per
parent benchmark to keep the appendix manageable.

Output: prints the LaTeX block to stdout; the caller redirects it into the
appendix.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from analysis.lib.loaders import PARENT_GROUPS


HERE = Path(__file__).resolve().parent
VERDICTS_DIR = HERE / "reports" / "verification" / "verdicts" / "search"
BANK = HERE / "reports" / "verification" / "flagged_bank.jsonl"

# Pretty parent labels for the heading.
PARENT_PRETTY = {
    "CTI": "CTI-Bench",
    "ATHENA": "AthenaBench",
    "SECURE": "SECURE",
    "REDSAGE": "RedSage-MCQ",
    "CYBERMETRIC": "CyberMetric",
    "MMLU-CS": "MMLU-CS",
    "SecBench": "SecBench",
    "SecEval": "SecEval",
    "SEVENLLM": "SEvenLLM",
}


def parent_of(task: str) -> str:
    for parent, members in PARENT_GROUPS:
        if task in members:
            return parent
    return "OTHER"


def load_bank_index() -> dict:
    """{(task, idx) -> bank record} so we can pull the original prompt."""
    out = {}
    if not BANK.exists():
        return out
    with open(BANK) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[(r["task"], str(r["index"]))] = r
    return out


def load_mislabel_examples() -> list[dict]:
    rows = []
    for f in VERDICTS_DIR.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("verdict") != "majority_correct":
            continue
        # need at least one citation
        cits = d.get("citations") or []
        if not cits:
            continue
        rows.append(d)
    return rows


def latex_escape(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
             .replace("#", "\\#").replace("_", "\\_").replace("{", "\\{")
             .replace("}", "\\}").replace("^", "\\^{}").replace("~", "\\~{}"))


def truncate_then_escape(s: str, n: int) -> str:
    """Trim to ~n chars on a word boundary, escape for LaTeX, then append the
    `\\ldots` literal (so it survives latex_escape)."""
    s = (s or "").strip().replace("\n", " ").replace("  ", " ")
    truncated = len(s) > n
    if truncated:
        s = s[: n - 1].rsplit(" ", 1)[0]
    s = latex_escape(s)
    if truncated:
        s += r"\ldots"
    return s


def render_example(task: str, rec: dict, bank: dict) -> str:
    idx = str(rec["index"])
    bank_rec = bank.get((task, idx), {})
    prompt = bank_rec.get("question", "") or rec.get("question", "")
    gold = rec.get("gold", "")
    pred = rec.get("majority_prediction", "")
    just = rec.get("justification", "")
    cits = rec.get("citations") or []
    af = rec.get("agreement_fraction", "")

    parent = parent_of(task)
    title = f"{PARENT_PRETTY.get(parent, parent)} / {latex_escape(task)} (idx {latex_escape(idx)}, $\\tau{{=}}{af}$)"

    # Body
    lines = []
    lines.append(r"\begin{tcolorbox}[")
    lines.append(r"  colback=red!4, colframe=red!55!black, boxrule=0.5pt,")
    lines.append(r"  arc=2pt, left=6pt, right=6pt, top=4pt, bottom=4pt,")
    lines.append(rf"  title={{\textbf{{{title}}}}},")
    lines.append(r"  fonttitle=\small\bfseries, coltitle=white,")
    lines.append(r"  colbacktitle=red!55!black, breakable]")
    lines.append(r"\small")
    lines.append(rf"\textbf{{Prompt:}} {truncate_then_escape(prompt, 600)}\\[2pt]")
    lines.append(rf"\textbf{{Published gold answer:}} \texttt{{{latex_escape(str(gold))}}}\\[2pt]")
    lines.append(rf"\textbf{{Model-majority prediction:}} \texttt{{{latex_escape(str(pred))}}}\\[2pt]")
    lines.append(rf"\textbf{{Search-grounded verdict:}} \emph{{{truncate_then_escape(just, 700)}}}\\[2pt]")
    lines.append(r"\textbf{Citations:}")
    lines.append(r"\begin{itemize}[leftmargin=1.4em,topsep=2pt,parsep=0pt,itemsep=1pt]")
    for c in cits[:3]:
        url = c.get("url", "")
        quote = truncate_then_escape(c.get("quote", ""), 200)
        lines.append(rf"  \item \href{{{url}}}{{\nolinkurl{{{url}}}}}")
        if quote:
            lines.append(rf"        \quad{{\footnotesize\textit{{``{quote}''}}}}")
    lines.append(r"\end{itemize}")
    lines.append(r"\end{tcolorbox}")
    lines.append("")
    return "\n".join(lines)


def main():
    bank = load_bank_index()
    examples = load_mislabel_examples()

    # Group by parent → task → records
    by_parent: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in examples:
        task = rec.get("task", "")
        if not task:
            continue
        by_parent[parent_of(task)][task].append(rec)

    print(r"\subsection{Confirmed gold-mislabel examples (sampled)}")
    print(r"\label{app:mislabel-examples}")
    print()
    print(r"For each benchmark family with at least one verifier-confirmed mislabel "
          r"(verdict \texttt{majority\_correct}), we render one representative "
          r"example below. Each box shows the original benchmark prompt (truncated), "
          r"the published gold answer, the majority of evaluated models' prediction, "
          r"the search-grounded verifier's reasoning, and the citations the verifier "
          r"used to ground its verdict (tier-1/2 cybersecurity sources only).")
    print()

    for parent, _ in PARENT_GROUPS:
        if parent not in by_parent:
            continue
        # pick one example per task in this parent, max 2 per parent total
        picks: list[tuple[str, dict]] = []
        for task in sorted(by_parent[parent]):
            recs = by_parent[parent][task]
            # prefer highest agreement_fraction, then index ascending for stability
            recs.sort(key=lambda r: (-(float(r.get("agreement_fraction", 0))),
                                      int(r.get("index", 0)) if str(r.get("index", "")).isdigit() else 0))
            picks.append((task, recs[0]))
            if len(picks) >= 2:
                break
        if not picks:
            continue
        print(rf"\paragraph{{{PARENT_PRETTY.get(parent, parent)}.}}")
        print()
        for task, rec in picks:
            print(render_example(task, rec, bank))


if __name__ == "__main__":
    main()
