"""Answer extractors lifted from analyze_gold_errors.py.

These convert raw model responses and gold strings into normalized predictions
suitable for set-/string-equality majority voting.
"""

from __future__ import annotations

import re
from typing import Optional


MCQ_TASKS = {
    "mcq", "cybermetric", "mmlu_cs", "mmlu-cs", "seceval", "secbench",
    "secure_maet", "secure_cwet",
    "ckt", "redsage_frameworks", "redsage_generals", "redsage_kali",
    "redsage_cli", "redsage_skills",
}

# True/False/X — secure_kcv uses single-letter T/F/X labels.
TFX_TASKS = {"secure_kcv"}

ID_TASKS = {"ate", "athena_ate", "rms"}

TEXT_TASKS = {"rcm", "athena_rcm", "taa"}

VSP_TASKS = {"vsp", "athena_vsp"}


_MCQ_RE = re.compile(r"\b([A-Da-d])\b")
_MCQ_GOLD_RE = re.compile(r"[A-Da-d]")
_TFX_RE = re.compile(r"\b([TFXtfx])\b")
_TECH_RE = re.compile(r"T\d{4}(?:\.\d{3})?", re.IGNORECASE)
_MIT_RE = re.compile(r"M\d{4}", re.IGNORECASE)
_CVSS_RE = re.compile(
    r"(?:CVSS:3\.[01]/)?AV:[A-Z]+/AC:[A-Z]+/PR:[A-Z]+/UI:[A-Z]+"
    r"/S:[A-Z]+/C:[A-Z]+/I:[A-Z]+/A:[A-Z]+",
    re.IGNORECASE,
)


def normalize_task(task: str) -> str:
    return task.replace("-", "_").lower()


def strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def extract_mcq_answer(response: str) -> str:
    response = strip_think(response)
    letters = sorted(set(_MCQ_RE.findall(response)))
    return "".join(l.upper() for l in letters) if letters else ""


def extract_tfx_answer(response: str) -> str:
    response = strip_think(response)
    matches = _TFX_RE.findall(response)
    if not matches:
        return ""
    # Last single TFX letter mentioned tends to be the final answer.
    return matches[-1].upper()


def extract_technique_ids(text: str) -> frozenset:
    return frozenset(m.upper() for m in _TECH_RE.findall(text))


def extract_mitigation_ids(text: str) -> frozenset:
    return frozenset(m.upper() for m in _MIT_RE.findall(text))


def extract_cvss_vector(text: str) -> str:
    text = strip_think(text)
    match = _CVSS_RE.search(text)
    return match.group(0).upper() if match else ""


def normalize_gold(gold: str, task: str) -> str:
    task = normalize_task(task)
    if task in MCQ_TASKS:
        letters = sorted(set(_MCQ_GOLD_RE.findall(gold)))
        return "".join(l.upper() for l in letters)
    if task in TFX_TASKS:
        g = gold.strip().upper()
        return g[0] if g and g[0] in {"T", "F", "X"} else ""
    if task in ID_TASKS:
        if "ate" in task:
            ids = extract_technique_ids(gold)
        else:
            ids = extract_mitigation_ids(gold)
        return str(sorted(ids))
    if task in VSP_TASKS:
        vec = extract_cvss_vector(gold)
        return vec
    return gold.strip()


def extract_prediction(response: str, task: str) -> Optional[str]:
    task = normalize_task(task)
    response = (response or "").strip()
    if not response:
        return None

    if task in MCQ_TASKS:
        pred = extract_mcq_answer(response)
        return pred if pred else None

    if task in TFX_TASKS:
        pred = extract_tfx_answer(response)
        return pred if pred else None

    if task in ID_TASKS:
        cleaned = strip_think(response)
        if "ate" in task:
            ids = extract_technique_ids(cleaned)
        else:
            ids = extract_mitigation_ids(cleaned)
        return str(sorted(ids)) if ids else None

    if task in VSP_TASKS:
        vec = extract_cvss_vector(response)
        return vec if vec else None

    if task in TEXT_TASKS:
        text = strip_think(response).lower().strip()
        return text[:200] if text else None

    return None


VOTABLE_TASKS = MCQ_TASKS | TFX_TASKS | ID_TASKS | VSP_TASKS
