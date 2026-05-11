"""Knowledge-vs-Analytical classifier prompt + few-shot pool + parser.

Two-class operational definition:

  Knowledge-oriented (K)
    The question is answered by retrieving a fact, term, definition, standard
    parameter, or procedural step from cybersecurity domain knowledge. Single-step
    factual recall. Typical surface form: short MCQ on definitions, properties,
    configuration values, RFC numbers.

  Analytical (A)
    The question requires reasoning over a scenario, structured extraction from
    text, multi-step inference, scoring, or attribution. Surface forms: CVSS-vector
    composition, MITRE ATT&CK technique extraction from narratives, CWE
    classification of vulnerability descriptions, threat-actor attribution.

Output envelope (strict JSON in a fenced ```json``` block):

  {
    "class": "K" | "A",
    "confidence": "low" | "medium" | "high",
    "rationale": "<one short sentence>"
  }
"""

from __future__ import annotations

import json
import re
from typing import Optional


# ----------------------- few-shot pool -----------------------
# Three K + three A examples. Drawn from sub-tasks the §2 taxonomy makes
# unambiguous (CYBERMETRIC → K-trivia; ATE → A-extraction; VSP → A-vector;
# MCQ → K-trivia; RCM → A-classification; SECEVAL → K).

_FEW_SHOT = [
    {
        "question": "Which of the following is a desirable property of a biometric system? Options: A) Permanent, B) Transferability, C) Uniformity, D) Forgiveness",
        "label": "K",
        "rationale": "Single-step factual recall about biometric-system design properties.",
    },
    {
        "question": "Which TCSEC level is labeled Controlled Access Protection? A. C1  B. C2  C. C3  D. B1",
        "label": "K",
        "rationale": "Definition lookup of a security-evaluation level.",
    },
    {
        "question": "What is the default value for the Windows Security policy 'Increase scheduling priority' user right?",
        "label": "K",
        "rationale": "Recall of a documented Windows policy default.",
    },
    {
        "question": "Given the following intrusion narrative, list the MITRE ATT&CK technique IDs observed. The attacker compromised a domain account via spearphishing, then authenticated to the VPN and used PowerShell to enumerate Active Directory.",
        "label": "A",
        "rationale": "Multi-step extraction of MITRE techniques from a free-form scenario.",
    },
    {
        "question": "Read the CVE description and produce the corresponding CVSS:3.1 vector string capturing the attack vector, complexity, privileges, user-interaction, scope, and confidentiality/integrity/availability impacts.",
        "label": "A",
        "rationale": "Structured CVSS vector composition by reasoning over the description.",
    },
    {
        "question": "Read the following vulnerability description and identify the most appropriate CWE category that explains the root cause. Description: An application copies data from one buffer into a fixed-size destination without checking bounds, allowing memory corruption when the input exceeds the buffer length.",
        "label": "A",
        "rationale": "Reasoning about root cause to assign a CWE class — not a name lookup.",
    },
]


# ----------------------- prompts -----------------------

SYS = """You are a careful annotator classifying cybersecurity benchmark questions into one of two classes:

K = Knowledge-oriented. Answered by retrieving a fact, definition, standard parameter,
    or procedural step from cybersecurity domain knowledge. Single-step factual recall.
    Surface form is usually a short MCQ on definitions, properties, configuration values,
    standards, or named entities.

A = Analytical. Requires reasoning over a scenario, structured extraction from text,
    multi-step inference, scoring, or attribution. Surface form usually involves a free-form
    description that the answerer must analyze (CVSS vector composition, MITRE ATT&CK
    technique extraction, CWE root-cause classification, threat-actor attribution).

Decision rules:
- If a single fact or definition produces the answer, choose K.
- If the answer requires reading a scenario / report / description and inferring or
  extracting structured information from it, choose A.
- Multi-choice format alone does not make a question K — what matters is whether the
  *answer* requires recall or reasoning.
- Choose A if the question presents a non-trivial input (code, log, CVE description,
  attack narrative) that must be analyzed before answering.

Always reply with strict JSON in a fenced ```json``` block, with exactly these keys:
"class" (K or A), "confidence" (low/medium/high), "rationale" (one short sentence).
"""


def _format_few_shot() -> str:
    parts = []
    for ex in _FEW_SHOT:
        parts.append(f"Question: {ex['question']}")
        parts.append("```json")
        parts.append(json.dumps(
            {"class": ex["label"], "confidence": "high", "rationale": ex["rationale"]}
        ))
        parts.append("```")
        parts.append("")
    return "\n".join(parts).strip()


def build_user_prompt(question: str) -> str:
    return (
        f"Here are six worked examples; classify the final question in the same format.\n\n"
        f"{_format_few_shot()}\n\n"
        f"Question: {question}\n"
    )


# ----------------------- parser -----------------------

_VERDICT_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_ANY_JSON_RE = re.compile(r"\{[^{}]*\"class\"[^{}]*\}", re.DOTALL)


def parse_verdict(raw_text: str) -> Optional[dict]:
    """Extract `{"class": K|A, "confidence": ..., "rationale": ...}` from model output."""
    if not raw_text:
        return None
    m = _VERDICT_RE.search(raw_text)
    if not m:
        m = _ANY_JSON_RE.search(raw_text)
    if not m:
        # last-ditch: scan for the bare class letter
        m2 = re.search(r"\b(K|A)\b", raw_text)
        if m2:
            return {"class": m2.group(1), "confidence": "low", "rationale": ""}
        return None
    try:
        d = json.loads(m.group(1) if "json" in raw_text and m.group(0).startswith("```") else m.group(0))
    except json.JSONDecodeError:
        return None
    cls = (d.get("class") or "").strip().upper()
    if cls not in {"K", "A"}:
        # Some models will say "Knowledge"/"Analytical"; map.
        if cls.startswith("K"):
            cls = "K"
        elif cls.startswith("A"):
            cls = "A"
        else:
            return None
    return {
        "class": cls,
        "confidence": (d.get("confidence") or "low").lower(),
        "rationale": (d.get("rationale") or "").strip(),
    }
