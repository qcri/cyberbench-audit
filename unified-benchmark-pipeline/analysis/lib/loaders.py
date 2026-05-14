"""Filesystem loaders for outputs/ judge and responses files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


# Repo layout assumption: caller passes outputs_root, defaults to ../outputs
OUTPUTS_DIRNAME = "outputs"


# 24 sub-tasks ordered for the master table
TASK_ORDER = [
    # CTI
    "ate", "rcm", "vsp", "ckt", "rms", "taa", "cti_taa", "mcq",
    # ATHENA
    "athena_ate", "athena_rcm", "athena_vsp",
    # SECURE
    "secure_maet", "secure_cwet", "secure_kcv",
    # REDSAGE
    "redsage_frameworks", "redsage_generals", "redsage_skills",
    "redsage_cli", "redsage_kali",
    # CYBERMETRIC
    "cybermetric",
    # Other standalone MCQ
    "mmlu_cs", "secbench", "seceval",
    # SEVENLLM (open-ended structured CTI extraction)
    "sevenllm",
]


PARENT_GROUPS = [
    # CTI-Bench upstream (RISys-Lab HF cti-* subsets + maveryn TSV cti-taa)
    ("CTI", ["mcq", "rcm", "vsp", "ate", "cti_taa"]),
    # AthenaBench JSONL (CKT/RMS + 100-item TAA + the 3 expanded ATE/RCM/VSP)
    ("ATHENA", ["ckt", "rms", "taa", "athena_ate", "athena_rcm", "athena_vsp"]),
    ("SECURE", ["secure_maet", "secure_cwet", "secure_kcv"]),
    ("REDSAGE", ["redsage_frameworks", "redsage_generals", "redsage_skills",
                 "redsage_cli", "redsage_kali"]),
    ("CYBERMETRIC", ["cybermetric"]),
    # The three previously-grouped "MCQ-Standalone" benchmarks are now reported
    # individually so reviewers can see their distinct K/A compositions
    # (MMLU-CS and SecBench are knowledge-heavy; SecEval is the most analytical).
    ("MMLU-CS", ["mmlu_cs"]),
    ("SecBench", ["secbench"]),
    ("SecEval", ["seceval"]),
    ("SEVENLLM", ["sevenllm"]),
]


# Judge versions known to the analysis pipeline. The directory suffix is the
# value; default judge has no suffix.
JUDGE_VERSIONS = ("", "_v1", "_v2")


# Default ordering — used to surface a stable list of models. Discovered at
# runtime from outputs/ but we prefer this canonical order when present.
PREFERRED_MODEL_ORDER = [
    "GPT-5.4",
    "Llama-3.3-70B-Instruct",
    "Llama-Primus-Nemotron-70B-Instruct",
    "Gemma-4-31B-it",
    "Qwen3.6-35B-A3B",
    "Fanar-2-27B-Instruct",
    "GPT-oss-20B",
    "Foundation-Sec-8B-Instruct",
    "Llama-Primus-Merged",
    "RedSage-Qwen3-8B-DPO",
]


def _strip_version_suffix(name: str) -> str:
    """Strip a known judge-version suffix from a model name."""
    for suf in ("_v2", "_v1"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def discover_models(outputs_root: Path) -> List[str]:
    """Find all models that have any judge_*/ or responses_*/ folder.

    Filters out:
      - smoke-test artifacts (suffix `_smoke`)
      - cross-judge comparison dirs (`judge_<MODEL>__by_<JUDGE_ALIAS>/`,
        produced by analysis/compare_judges.py).
    """
    seen = set()
    for child in outputs_root.iterdir():
        name = child.name
        for prefix in ("judge_", "responses_"):
            if name.startswith(prefix):
                rest = name[len(prefix):]
                if rest.endswith("_smoke"):
                    break
                if "__by_" in rest:
                    # Cross-judge comparison output, not a real model
                    break
                rest = _strip_version_suffix(rest)
                seen.add(rest)
                break
    ordered = [m for m in PREFERRED_MODEL_ORDER if m in seen]
    extras = sorted(seen - set(ordered))
    return ordered + extras


def _resolve_version(version: Optional[str], v1: bool) -> str:
    """Normalise the version arg. `version=` wins; `v1=True` is a legacy alias."""
    if version is None:
        return "_v1" if v1 else ""
    if version and not version.startswith("_"):
        version = "_" + version
    return version


def detailed_path(outputs_root: Path, model: str, task: str,
                  v1: bool = False, version: Optional[str] = None) -> Path:
    suffix = _resolve_version(version, v1)
    return outputs_root / f"judge_{model}{suffix}" / "eval_results" / f"{task}_detailed.jsonl"


def responses_path(outputs_root: Path, model: str, task: str,
                   version: Optional[str] = None) -> Path:
    """Locate the inference responses file for (model, task).

    `version` lets callers point at the v2 inputs (responses_<MODEL>_v2/) when
    needed for cross-checking; default is the local responses_<MODEL>/ dir.
    """
    suffix = _resolve_version(version, False)
    base = outputs_root / f"responses_{model}{suffix}"
    # task names in responses files use hyphens for mmlu-cs only; try both.
    p1 = base / f"{task}_responses.jsonl"
    if p1.exists():
        return p1
    p2 = base / f"{task.replace('_', '-')}_responses.jsonl"
    return p2


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_detailed(outputs_root: Path, model: str, task: str) -> Tuple[Optional[List[dict]], Optional[str]]:
    """Return (records, source) where source is 'judge', 'judge_v1', 'judge_v2', or None.

    Prefers default judge; falls back through v1 then v2.
    """
    for version, label in [("", "judge"), ("_v1", "judge_v1"), ("_v2", "judge_v2")]:
        path = detailed_path(outputs_root, model, task, version=version)
        if path.exists():
            records = list(iter_jsonl(path))
            if records:
                return records, label
    return None, None


def load_detailed_with_version(outputs_root: Path, model: str, task: str,
                               version: str) -> Optional[List[dict]]:
    """Load per-sample records from one specific judge version (or None if missing).

    `version` should be one of "" (default), "_v1", "_v2" (or "v1" / "v2" for
    convenience — the leading underscore is added if absent).
    """
    path = detailed_path(outputs_root, model, task, version=version)
    if not path.exists():
        return None
    records = list(iter_jsonl(path))
    return records or None


def summary_path(outputs_root: Path, model: str,
                 v1: bool = False, version: Optional[str] = None) -> Path:
    suffix = _resolve_version(version, v1)
    return outputs_root / f"judge_{model}{suffix}" / "eval_results" / "summary.json"


def load_summary_accuracy(outputs_root: Path, model: str, task: str,
                          v1: bool = False, version: Optional[str] = None) -> Optional[float]:
    """Read accuracy for a task from summary.json (key is task.upper())."""
    path = summary_path(outputs_root, model, version=_resolve_version(version, v1))
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    tasks = data.get("tasks", {})
    key = task.replace("-", "_").upper()
    entry = tasks.get(key)
    if entry is None:
        return None
    return entry.get("accuracy")


def load_summary_n(outputs_root: Path, model: str, task: str,
                   v1: bool = False, version: Optional[str] = None) -> int:
    path = summary_path(outputs_root, model, version=_resolve_version(version, v1))
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return 0
    entry = data.get("tasks", {}).get(task.replace("-", "_").upper(), {})
    return int(entry.get("total", 0) or 0)


def load_summary_entry(outputs_root: Path, model: str, task: str,
                       v1: bool = False, version: Optional[str] = None) -> Optional[dict]:
    """Return the full task entry from summary.json, or None if missing."""
    path = summary_path(outputs_root, model, version=_resolve_version(version, v1))
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    entry = data.get("tasks", {}).get(task.replace("-", "_").upper())
    return entry


def load_responses(outputs_root: Path, model: str, task: str) -> Dict[str, dict]:
    """Load responses keyed by string sample index."""
    path = responses_path(outputs_root, model, task)
    out = {}
    for r in iter_jsonl(path):
        idx = str(r.get("index", ""))
        if idx:
            out[idx] = r
    return out
