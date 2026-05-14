"""Two GPT-5.4 verifier harnesses for flagged-sample triage.

`agent_search`  — Azure Responses API + `web_search_preview` tool. The system
                  prompt restricts citations to a tiered cybersec whitelist;
                  citations outside the whitelist demote the verdict to
                  `uncertain`.
`agent_direct`  — Azure Chat Completions API, no tools. Same output schema.

Both return a dict:
    {
      "verdict": "gold_correct" | "majority_correct" | "both_wrong" | "uncertain",
      "confidence": "low" | "medium" | "high",
      "justification": str,
      "citations": [{"url": str, "tier": int, "quote": str}]   # search agent only
    }
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


AZURE_ENDPOINT = "https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/"
AZURE_API_VERSION = "2025-04-01-preview"
MODEL = "gpt-5.4"


# ----------------- domain whitelist -----------------

WHITELIST_TIER1 = [
    "attack.mitre.org", "cve.mitre.org", "cwe.mitre.org", "capec.mitre.org",
    "nvd.nist.gov", "csrc.nist.gov", "nist.gov/itl",
    "cisa.gov", "us-cert.cisa.gov", "iana.org",
    "datatracker.ietf.org", "rfc-editor.org",
    "first.org", "owasp.org",
]
# Tier 2: vendor primary docs / security advisories that are authoritative for
# their own product surface (Windows behaviour, macOS behaviour, etc.)
WHITELIST_TIER2 = [
    "learn.microsoft.com", "docs.microsoft.com",
    "msrc.microsoft.com", "microsoft.com/security",
    "support.apple.com", "developer.apple.com",
    "oracle.com/security-alerts", "docs.oracle.com",
    "redhat.com/security", "access.redhat.com/security",
    "kali.org/docs", "offensive-security.com",
    "pcisecuritystandards.org",
    "developer.mozilla.org",  # MDN — authoritative on web/HTTP semantics
    "ietf.org",
    "tools.ietf.org",
    "wireshark.org",
]


def domain_tier(url: str) -> int:
    if not url:
        return 3
    u = url.lower()
    for d in WHITELIST_TIER1:
        if d in u:
            return 1
    for d in WHITELIST_TIER2:
        if d in u:
            return 2
    return 3


# ----------------- prompts -----------------

SYS_SEARCH = """You are an expert cybersecurity reviewer auditing benchmark
labels. You receive a multiple-choice question, a gold answer, and a model-
majority prediction that disagrees with the gold. Your job is to decide which
of the two is correct.

Use the web search tool. Prefer authoritative sources only:
- Tier 1 (always preferred): MITRE ATT&CK, CVE, CWE, CAPEC, NIST NVD/CSRC, CISA,
  IANA, IETF RFCs, FIRST.org, OWASP.
- Tier 2 (acceptable when Tier 1 is silent): vendor security advisories
  (Microsoft, Apple, Oracle, Red Hat), Kali docs, PCI standards.
- Do NOT cite blog posts, exam-prep sites, or opinion content.

Output STRICT JSON in a fenced ```json``` block, with this schema:
{
  "verdict": "gold_correct" | "majority_correct" | "both_wrong" | "uncertain",
  "confidence": "low" | "medium" | "high",
  "justification": "<one paragraph explaining the decision>",
  "citations": [
    {"url": "<full URL>", "quote": "<short verbatim quote from the source>"}
  ]
}

Rules:
- "gold_correct" means the gold answer is right and the models were wrong.
- "majority_correct" means the gold answer is mislabelled and the models are right.
- "both_wrong" means neither answer is correct (a third option would be).
- Use "uncertain" if you cannot find authoritative grounding within 3-5 searches.
- Always cite at least one Tier-1 or Tier-2 source unless the verdict is "uncertain".
"""


SYS_DIRECT = """You are an expert cybersecurity reviewer auditing benchmark
labels. You receive a multiple-choice question, a gold answer, and a model-
majority prediction that disagrees with the gold. Decide which is correct
based on your own knowledge.

Output STRICT JSON in a fenced ```json``` block, with this schema:
{
  "verdict": "gold_correct" | "majority_correct" | "both_wrong" | "uncertain",
  "confidence": "low" | "medium" | "high",
  "justification": "<one paragraph explaining the decision>"
}

- "gold_correct" means the gold answer is right and the models were wrong.
- "majority_correct" means the gold answer is mislabelled and the models are right.
- "both_wrong" means neither answer is correct.
- "uncertain" if you cannot decide with reasonable confidence.
"""


def build_user_prompt(record: dict) -> str:
    """Format the flagged sample into the prompt body."""
    lines = [
        f"Task: {record['task']}",
        f"Sample index: {record['index']}",
        f"Models that agreed against the gold: {record['n_models_with_data']}"
        f" with agreement fraction {record['agreement_fraction']:.3f}",
        "",
        "==== Question ====",
        record.get("question") or "(question text unavailable)",
        "",
        f"==== Gold answer (flagged as suspect) ====",
        record["gold"],
        "",
        f"==== Model-majority prediction ====",
        record["majority_prediction"],
    ]
    return "\n".join(lines)


# ----------------- API plumbing -----------------

def _post_responses(body: dict, key: str, retries: int = 3, timeout: int = 90) -> dict:
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/responses?api-version={AZURE_API_VERSION}"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={"api-key": key, "Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=timeout)
            return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode()[:300]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(last_err or "unknown error")


def _extract_text_and_citations(resp: dict):
    """Walk the Responses-API output to extract assistant text + citations."""
    text_parts = []
    citations = []
    for msg in resp.get("output", []) or []:
        if msg.get("type") == "message":
            for c in msg.get("content", []) or []:
                if c.get("text"):
                    text_parts.append(c["text"])
                for ann in c.get("annotations", []) or []:
                    url = ann.get("url")
                    if url:
                        citations.append({"url": url})
    return "\n".join(text_parts), citations


def _parse_json_block(text: str) -> Optional[dict]:
    """Pull the first ```json ... ``` block; fall back to first {...}."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ----------------- public verifiers -----------------

def verify_with_search(record: dict, key: str) -> dict:
    body = {
        "model": MODEL,
        "instructions": SYS_SEARCH,
        "input": build_user_prompt(record),
        "tools": [{"type": "web_search_preview"}],
    }
    resp = _post_responses(body, key)
    text, citations = _extract_text_and_citations(resp)
    parsed = _parse_json_block(text) or {}

    # tier each citation
    cited = parsed.get("citations") or []
    if not cited and citations:
        cited = [{"url": c["url"]} for c in citations]
    for c in cited:
        c["tier"] = domain_tier(c.get("url", ""))

    verdict = parsed.get("verdict", "uncertain")
    if verdict in {"gold_correct", "majority_correct", "both_wrong"}:
        if not any(c.get("tier", 3) <= 2 for c in cited):
            verdict = "uncertain"
            parsed.setdefault("justification", "")
            parsed["justification"] = (
                "[downgraded: no Tier-1/Tier-2 citation] "
                + parsed.get("justification", "")
            )

    return {
        "verdict": verdict,
        "confidence": parsed.get("confidence", "low"),
        "justification": parsed.get("justification", text[:600]),
        "citations": cited,
        "raw_text": text,
    }


def verify_direct(record: dict, key: str) -> dict:
    """Plain chat-completions call, no tools."""
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint=AZURE_ENDPOINT,
        api_key=key,
    )
    last_err = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYS_DIRECT},
                    {"role": "user", "content": build_user_prompt(record)},
                ],
                max_completion_tokens=1024,
                temperature=0.0,
            )
            text = (r.choices[0].message.content or "").strip()
            parsed = _parse_json_block(text) or {}
            return {
                "verdict": parsed.get("verdict", "uncertain"),
                "confidence": parsed.get("confidence", "low"),
                "justification": parsed.get("justification", text[:600]),
                "citations": [],
                "raw_text": text,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < 2:
                time.sleep(2 ** attempt)
    return {
        "verdict": "uncertain",
        "confidence": "low",
        "justification": f"[error after retries] {last_err}",
        "citations": [],
        "raw_text": "",
    }
