"""Re-parse cached Qwen3.6-35B-A3B raw_text outputs with a smarter parser.

Failure mode (see secure_kcv_107, secure_kcv_138, ckt_*):
  Qwen's verbose chain-of-thought blows through max_completion_tokens=400 before
  emitting the final JSON envelope. The original parser falls back to
  `\\b(K|A)\\b` and matches the FIRST occurrence -- which is always K because the
  prompt structure introduces "K (Knowledge-oriented) or A (Analytical)".

  Fix: prefer (a) explicit JSON envelope, (b) "class": "X" / "Class: X" /
  "Verdict: X" patterns, (c) declarative phrases ("fits A perfectly", "is appropriate"),
  (d) LAST `\\b(K|A)\\b` occurrence in the trailing 400 chars (model concludes at end).
"""

import json, re, sys
from pathlib import Path

VERDICTS = Path("/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/analysis/reports/coverage/verdicts/Qwen3.6-35B-A3B")

JSON_FENCED = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
JSON_ANY    = re.compile(r"\{[^{}]*\"class\"[^{}]*\}", re.DOTALL)
KEY_VAL     = re.compile(r"\"class\"\s*:\s*\"?([KA])\"?", re.IGNORECASE)
EXPLICIT    = re.compile(r"\b(?:class|verdict|final\s+answer|answer)\s*[:=]\s*\"?\*?\*?([KA])\*?\*?\"?", re.IGNORECASE)
DECLARATIVE = re.compile(
    r"(?:fits|should\s+be|is)\s+\*{0,2}([KA])\*{0,2}\b",
    re.IGNORECASE,
)
BARE        = re.compile(r"\b([KA])\b")


def smart_parse(rt: str):
    if not rt:
        return None, "empty"

    # 1. Strict JSON envelope
    m = JSON_FENCED.search(rt) or JSON_ANY.search(rt)
    if m:
        try:
            d = json.loads(m.group(1) if m.re is JSON_FENCED else m.group(0))
            cls = (d.get("class") or "").strip().upper()
            if cls.startswith("K"): return "K", "json_envelope"
            if cls.startswith("A"): return "A", "json_envelope"
        except json.JSONDecodeError:
            pass

    # 2. "class": "X" key/value occurring anywhere (but take LAST match — model
    #    often restates the schema at start, conclusion is later)
    matches = list(KEY_VAL.finditer(rt))
    if matches:
        return matches[-1].group(1).upper(), "key_val_last"

    # 3. Class:/Verdict:/Final answer: prefix, take last
    matches = list(EXPLICIT.finditer(rt))
    if matches:
        return matches[-1].group(1).upper(), "explicit_last"

    # 4. Declarative tail: "fits A", "should be K", "is K" — take last
    matches = list(DECLARATIVE.finditer(rt))
    if matches:
        return matches[-1].group(1).upper(), "declarative_last"

    # 5. Last bare K|A in the trailing 400 chars (model usually concludes at end;
    #    avoids the spurious K from "K (Knowledge-oriented) or A (Analytical)" preamble)
    tail = rt[-400:]
    matches = list(BARE.finditer(tail))
    if matches:
        return matches[-1].group(1).upper(), "bare_tail_last"

    # 6. Last bare K|A overall
    matches = list(BARE.finditer(rt))
    if matches:
        return matches[-1].group(1).upper(), "bare_full_last"

    return None, "no_match"


# scan all Qwen verdicts
counts = {"changed": 0, "same": 0, "still_unparseable": 0}
modes = {}
flips = []

paths = sorted(VERDICTS.glob("*.json"))
print(f"reparsing {len(paths)} Qwen verdicts...")
for p in paths:
    d = json.loads(p.read_text())
    old = d.get("class")
    new, mode = smart_parse(d.get("raw_text", ""))
    modes[mode] = modes.get(mode, 0) + 1
    if new is None:
        counts["still_unparseable"] += 1
    elif new != old:
        counts["changed"] += 1
        flips.append((p.name, old, new, mode))
    else:
        counts["same"] += 1

print(f"\n# Re-parse summary (no model changes, just smarter post-hoc parsing):")
for k, v in counts.items():
    print(f"  {k}: {v}")
print(f"\n# Mode distribution:")
for m, n in sorted(modes.items(), key=lambda x: -x[1]):
    print(f"  {m}: {n}")
print(f"\n# Sample flips (old → new, mode):")
for f in flips[:15]:
    print(f"  {f}")

# Write the new parses (don't yet apply — print preview)
print(f"\n# WRITE? Apply to disk? Set --apply to commit.")

if "--apply" in sys.argv:
    for p in paths:
        d = json.loads(p.read_text())
        new, mode = smart_parse(d.get("raw_text", ""))
        d["class"] = new
        d["_parse_mode"] = mode
        p.write_text(json.dumps(d))
    print(f"  wrote updated 'class' field to {len(paths)} verdict files.")

