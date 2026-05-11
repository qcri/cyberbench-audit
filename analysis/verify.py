"""Run GPT-5.4 verifiers on the flagged-sample bank.

Two modes:
  --agent search     # web-search grounded
  --agent direct     # no tools, model knowledge only
  --agent both       # run both sequentially

Verdicts are cached per (task, idx, agent) under verification/verdicts/<agent>/.
Re-running skips already-cached verdicts.

Threshold-aware: --max-threshold 1.0 processes only the 84 unanimous samples
first; subsequent calls with lower thresholds (0.90, 0.833, 0.75) reuse those
verdicts and only spend API budget on the new samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from analysis.lib.verify import verify_direct, verify_with_search


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ENV_PATH = PROJECT_ROOT / ".env"
REPORTS = HERE / "reports"
VERIF_DIR = REPORTS / "verification"
BANK_PATH = VERIF_DIR / "flagged_bank.jsonl"


def load_env():
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def cache_path(agent: str, task: str, idx: str) -> Path:
    return VERIF_DIR / "verdicts" / agent / f"{task}_{idx}.json"


def load_bank():
    return [json.loads(l) for l in BANK_PATH.open() if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["search", "direct", "both"], default="both")
    p.add_argument("--max-threshold", type=float, default=1.0,
                   help="Only verify samples with first_threshold_at_or_above >= this value.")
    p.add_argument("--time-budget-s", type=float, default=180.0,
                   help="Stop after this many seconds; resume by re-running.")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N new verdicts in this run (0 = unlimited).")
    args = p.parse_args()

    load_env()
    key = os.environ.get("AZURE_API_KEY")
    if not key:
        print("AZURE_API_KEY not set", file=sys.stderr); sys.exit(1)

    bank = load_bank()
    targets = [r for r in bank if r["first_threshold_at_or_above"] >= args.max_threshold - 1e-9]
    print(f"target pool ({len(targets)}/{len(bank)} samples at threshold >= {args.max_threshold})")

    agents = ["search", "direct"] if args.agent == "both" else [args.agent]
    fns = {"search": verify_with_search, "direct": verify_direct}

    t0 = time.time()
    new_verdicts = 0
    for agent in agents:
        (VERIF_DIR / "verdicts" / agent).mkdir(parents=True, exist_ok=True)

    for rec in targets:
        if args.limit and new_verdicts >= args.limit:
            break
        if (time.time() - t0) > args.time_budget_s:
            print(f"time budget exceeded; resume by re-running. processed {new_verdicts} this run.")
            break
        for agent in agents:
            cp = cache_path(agent, rec["task"], rec["index"])
            if cp.exists():
                continue
            try:
                v = fns[agent](rec, key)
            except Exception as e:
                v = {
                    "verdict": "uncertain",
                    "confidence": "low",
                    "justification": f"[error] {type(e).__name__}: {e}",
                    "citations": [],
                    "raw_text": "",
                }
            v.update({
                "task": rec["task"],
                "index": rec["index"],
                "gold": rec["gold"],
                "majority_prediction": rec["majority_prediction"],
                "first_threshold_at_or_above": rec["first_threshold_at_or_above"],
                "agreement_fraction": rec["agreement_fraction"],
                "agent": agent,
            })
            cp.write_text(json.dumps(v))
            new_verdicts += 1
            if args.limit and new_verdicts >= args.limit:
                break

    print(f"new verdicts this run: {new_verdicts}")


if __name__ == "__main__":
    main()
