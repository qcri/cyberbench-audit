"""Re-tier search-agent verdicts after a whitelist update.

For every cached search-agent verdict whose verdict was downgraded to
`uncertain` because of an over-strict whitelist, re-evaluate the citation
tiers using the current `domain_tier` rules. If at least one citation is now
Tier 1 or Tier 2 AND the original (pre-downgrade) verdict was something
other than `uncertain`, restore that verdict.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from analysis.lib.verify import domain_tier


VERIF_DIR = Path(__file__).resolve().parent / "reports" / "verification"
SEARCH_DIR = VERIF_DIR / "verdicts" / "search"


def parse_original_verdict(raw_text: str):
    """Pull the verdict back out of the raw model JSON, if available."""
    if not raw_text:
        return None, None
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if not m:
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not m:
        return None, None
    try:
        d = json.loads(m.group(1) if "json" in raw_text else m.group(0))
    except json.JSONDecodeError:
        return None, None
    return d.get("verdict"), d.get("justification")


def main():
    fixed = 0
    seen = 0
    for path in SEARCH_DIR.glob("*.json"):
        try:
            txt = path.read_text()
            if not txt.strip():
                continue
            d = json.loads(txt)
        except (json.JSONDecodeError, OSError):
            continue
        seen += 1
        cits = d.get("citations") or []
        # always re-tier
        for c in cits:
            c["tier"] = domain_tier(c.get("url", ""))

        if d.get("verdict") != "uncertain":
            d["citations"] = cits
            path.write_text(json.dumps(d))
            continue

        # Check if it had been downgraded by us (look for our marker)
        just = d.get("justification") or ""
        if "[downgraded: no Tier-1/Tier-2 citation]" not in just:
            d["citations"] = cits
            path.write_text(json.dumps(d))
            continue

        # See if any citation is now Tier 1/2
        if not any(c.get("tier", 3) <= 2 for c in cits):
            d["citations"] = cits
            path.write_text(json.dumps(d))
            continue

        # Try to recover the original verdict from raw_text
        orig_verdict, orig_just = parse_original_verdict(d.get("raw_text", ""))
        if orig_verdict and orig_verdict != "uncertain":
            d["verdict"] = orig_verdict
            if orig_just:
                d["justification"] = orig_just
            d["citations"] = cits
            path.write_text(json.dumps(d))
            fixed += 1
        else:
            d["citations"] = cits
            path.write_text(json.dumps(d))

    print(f"scanned {seen} search verdicts; promoted {fixed} from 'uncertain' "
          f"back to original verdict via the relaxed whitelist")


if __name__ == "__main__":
    main()
