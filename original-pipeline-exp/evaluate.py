#!/usr/bin/env python3
"""
Evaluate fine-tuned model on multiple cybersecurity benchmarks

Supported benchmarks:
- CTI-Bench: MCQ, RCM, VSP, ATE (4 tasks)
- CyberMetric-500: Cybersecurity knowledge questions
- SecEval: 2126 cybersecurity knowledge questions
- CISSP: Cybersecurity certification questions

Supports both local model inference and API endpoint evaluation.
"""

import os
import re
import json
import time
import requests
from collections import Counter

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# Heavy ML imports — deferred; only loaded when local/API inference is actually needed.
# Do NOT import torch/transformers/datasets at module level — they may be unavailable
# or slow to initialize (e.g. CUDA init) in collected-eval environments.
torch = None
HAS_TORCH = False
load_dataset = None
HAS_DATASETS = False
AutoModelForCausalLM = None
AutoTokenizer = None
HAS_TRANSFORMERS = False
HAS_PEFT = False


def _ensure_inference_deps():
    """Lazily import heavy ML deps — called only from inference code paths."""
    global torch, HAS_TORCH, load_dataset, HAS_DATASETS
    global AutoModelForCausalLM, AutoTokenizer, HAS_TRANSFORMERS
    global HAS_PEFT
    if not HAS_TORCH:
        try:
            import torch as _torch
            torch = _torch
            HAS_TORCH = True
        except ImportError:
            pass
    if not HAS_DATASETS:
        try:
            from datasets import load_dataset as _ld
            load_dataset = _ld
            HAS_DATASETS = True
        except ImportError:
            pass
    if not HAS_TRANSFORMERS:
        try:
            from transformers import AutoModelForCausalLM as _amc, AutoTokenizer as _at
            AutoModelForCausalLM = _amc
            AutoTokenizer = _at
            HAS_TRANSFORMERS = True
        except ImportError:
            pass
    if not HAS_PEFT:
        try:
            from peft import PeftModel  # noqa: F401
            HAS_PEFT = True
        except ImportError:
            pass

import numpy as np

try:
    from sklearn.metrics import f1_score, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    def f1_score(*args, **kwargs):
        raise ImportError("scikit-learn is required for f1_score")
    def accuracy_score(*args, **kwargs):
        raise ImportError("scikit-learn is required for accuracy_score")

try:
    from cvss import CVSS3
    HAS_CVSS = True
except ImportError:
    CVSS3 = None
    HAS_CVSS = False


# Helpers: 
# ---------------------------------------------------------------------
# CTI-Bench faithful scoring helpers
# ---------------------------------------------------------------------

def normalize_ctibench_text(value) -> str:
    return str(value or "").strip()

# ---------------------------------------------------------------------
# CTI-Bench original formatting helpers
# From maveryn/cti-bench evaluation/model-prediction.ipynb
# ---------------------------------------------------------------------

def ctibench_format_mcq(text: str) -> str:
    text = str(text or "")
    lines = text.split("\n")

    last_line = lines[-1].rstrip()

    if last_line.startswith("A)") or last_line.startswith("B)") or last_line.startswith("C)") or last_line.startswith("D)"):
        return last_line[0]

    if last_line.endswith("A") or last_line.endswith("B") or last_line.endswith("C") or last_line.endswith("D"):
        return last_line[-1]

    if last_line.endswith("**") and len(last_line) >= 3:
        return last_line[-3]

    if len(last_line) == 0 and len(lines) >= 2:
        last_line = lines[-2].rstrip()

        if last_line.startswith("A)") or last_line.startswith("B)") or last_line.startswith("C)") or last_line.startswith("D)"):
            return last_line[0]

        if last_line.endswith("A") or last_line.endswith("B") or last_line.endswith("C") or last_line.endswith("D"):
            return last_line[-1]

        if last_line.endswith("**") and len(last_line) >= 3:
            return last_line[-3]

    return " ".join(text.split("\n"))


def ctibench_format_rcm(text: str):
    text = str(text or "")
    cwe_pattern = r"CWE-\d+"
    matches = re.findall(cwe_pattern, text)

    if matches:
        return matches[-1], True
    return text, False


def ctibench_format_vsp(text: str):
    text = str(text or "")
    cvss_pattern = (
        r"AV:[A-Za-z]+/AC:[A-Za-z]+/PR:[A-Za-z]+/"
        r"UI:[A-Za-z]+/S:[A-Za-z]+/C:[A-Za-z]+/"
        r"I:[A-Za-z]+/A:[A-Za-z]+"
    )
    matches = re.findall(cvss_pattern, text)

    if matches:
        return matches[-1], True
    return text, False


def ctibench_format_taa(text: str) -> str:
    """Original CTI-Bench format_taa from model-prediction.ipynb.

    The original notebook joins all lines with spaces and notes that manual
    extraction was required ("need to manually extract the attribution").
    The responses TSV used during evaluation already contains the manually
    extracted name, not the full response. We preserve the original logic;
    CTI-TAA cannot be scored faithfully from raw responses without a manual
    extraction step.
    """
    text = str(text or "")
    return " ".join(text.split("\n"))


def ctibench_format_ate(text: str) -> set:
    """Extract main MITRE ATT&CK technique IDs from a CTI-ATE response.

    Faithful to the prompt: "Ensure the final line contains only the IDs for the
    main techniques, separated by commas, excluding any subtechnique IDs."

    Sub-techniques are stripped via .split(".")[0], consistent with AthenaBench ATE.
    Last line is tried first; falls back to full-text scan.
    """
    text = str(text or "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    tid_re = re.compile(r'\bT\d{4}(?:\.\d+)?\b', re.IGNORECASE)

    def extract_ids(s: str) -> set:
        return {t.split(".")[0].upper() for t in tid_re.findall(s)}

    if lines:
        ids = extract_ids(lines[-1])
        if ids:
            return ids

    return extract_ids(text)


def score_ctibench_mcq(pred: str, gold: str) -> dict:
    """Faithful CTI-MCQ scoring.

    Original logic:
    - pred = row[col].upper()
    - gt = row["GT"].upper()
    - valid only if pred in ["A", "B", "C", "D", "X"]
    - correct if pred == gt
    - invalid predictions are excluded from total
    """
    pred_norm = normalize_ctibench_text(pred).upper()
    gold_norm = normalize_ctibench_text(gold).upper()

    valid = pred_norm in {"A", "B", "C", "D", "X"}
    correct = valid and pred_norm == gold_norm

    return {
        "pred": pred_norm,
        "gold": gold_norm,
        "valid": valid,
        "correct": correct,
        "score": 1.0 if correct else 0.0,
    }


def score_ctibench_rcm(pred: str, gold: str) -> dict:
    """Faithful CTI-RCM / CTI-RCM-2021 scoring.

    Original logic:
    - pred = row[col].upper()
    - gt = row["GT"].upper()
    - valid only if pred.startswith("CWE-")
    - correct if pred == gt
    - invalid predictions are excluded from total
    """
    pred_norm = normalize_ctibench_text(pred).upper()
    gold_norm = normalize_ctibench_text(gold).upper()

    valid = pred_norm.startswith("CWE-")
    correct = valid and pred_norm == gold_norm

    return {
        "pred": pred_norm,
        "gold": gold_norm,
        "valid": valid,
        "correct": correct,
        "score": 1.0 if correct else 0.0,
    }


def get_ctibench_cvss_score(cvss_vector: str) -> float:
    if not HAS_CVSS:
        raise ImportError("CTI-Bench VSP scoring requires: pip install cvss")
    return CVSS3(cvss_vector).scores()[0]


def score_ctibench_vsp(pred: str, gold: str, cvss_prefix: str = "CVSS:3.0/") -> dict:
    """Faithful CTI-VSP scoring.

    Original evaluation.ipynb logic:
    - pred = row[col].upper()
    - gt = row["GT"].upper()
    - pred_vector = cvss_prefix + pred
    - pred_score = CVSS3(pred_vector).scores()[0]
    - gt_score = CVSS3(gt).scores()[0]
    - invalid predictions are skipped
    """
    pred_norm = normalize_ctibench_text(pred).upper()
    gold_norm = normalize_ctibench_text(gold).upper()

    try:
        pred_vector = cvss_prefix + pred_norm
        pred_score = get_ctibench_cvss_score(pred_vector)
        gold_score = get_ctibench_cvss_score(gold_norm)
        error = abs(pred_score - gold_score)

        return {
            "pred": pred_norm,
            "gold": gold_norm,
            "valid": True,
            "correct": None,
            "score": None,
            "pred_vector": pred_vector,
            "pred_score": pred_score,
            "gold_score": gold_score,
            "error": error,
        }
    except Exception as e:
        return {
            "pred": pred_norm,
            "gold": gold_norm,
            "valid": False,
            "correct": None,
            "score": None,
            "pred_vector": cvss_prefix + pred_norm,
            "pred_score": None,
            "gold_score": None,
            "error": None,
            "error_message": str(e),
        }


def normalize_actor_name(actor: str) -> str:
    return normalize_ctibench_text(actor).lower()


def normalize_connection_dict(d: dict) -> dict:
    return {
        normalize_actor_name(k): [normalize_actor_name(v) for v in vals]
        for k, vals in d.items()
    }


def make_bidirectional_dict(d: dict) -> dict:
    """Mirror CTI-Bench notebook behavior by adding reverse edges."""
    out = {k: list(vs) for k, vs in d.items()}
    for actor in list(out):
        for linked_actor in list(out[actor]):
            out.setdefault(linked_actor, [])
            if actor not in out[linked_actor]:
                out[linked_actor].append(actor)
    return out


def is_ctibench_alias_connected(actor1: str, actor2: str, alias_dict: dict) -> bool:
    visited = set()
    queue = [actor1]

    while queue:
        current_actor = queue.pop(0)
        visited.add(current_actor)

        for alias in alias_dict.get(current_actor, []):
            if alias == actor2:
                return True
            if alias not in visited:
                queue.append(alias)

    return False


def is_ctibench_related_connected(actor1: str, actor2: str, alias_dict: dict, related_dict: dict) -> bool:
    visited = set()
    queue = [actor1]

    while queue:
        current_actor = queue.pop(0)
        visited.add(current_actor)

        for alias in alias_dict.get(current_actor, []):
            if alias == actor2:
                return True
            if alias not in visited:
                queue.append(alias)

        for related_actor in related_dict.get(current_actor, []):
            if related_actor == actor2:
                return True
            if related_actor not in visited:
                queue.append(related_actor)

    return False


def threat_actor_connection_ctibench(actor1: str, actor2: str, alias_dict: dict, related_dict: dict) -> str:
    """Faithful CTI-TAA connection logic.

    Returns:
    - "C" if connected through aliases
    - "P" if connected through aliases/related groups
    - "I" otherwise
    """
    actor1 = normalize_actor_name(actor1)
    actor2 = normalize_actor_name(actor2)

    alias_dict = make_bidirectional_dict(normalize_connection_dict(alias_dict))
    related_dict = make_bidirectional_dict(normalize_connection_dict(related_dict))

    if is_ctibench_alias_connected(actor1, actor2, alias_dict):
        return "C"

    if is_ctibench_related_connected(actor1, actor2, alias_dict, related_dict):
        return "P"

    return "I"


def extract_actor_from_response(text: str, alias_dict: dict, related_dict: dict) -> str:
    """Extract a threat actor name from a verbose model response.

    The original CTI-Bench evaluation/responses/cti-taa-responses.tsv shows
    that predictions are clean actor names (e.g. "APT28", "Lazarus", "X").
    The original authors extracted these manually from model outputs.

    This function automates that extraction: it scans all known actor names
    (keys + values from alias_dict and related_dict) against the normalized
    response text and returns the longest substring match — the most specific
    actor name the model mentioned.

    If no known actor name is found in the response, returns "" (equivalent
    to the original "X" — no valid prediction), which the BFS will score as
    incorrect.
    """
    text_norm = normalize_actor_name(text)

    # Build the full vocabulary of known actor names from both dicts
    all_names: set[str] = set()
    for d in (alias_dict, related_dict):
        for k, vs in d.items():
            name = normalize_actor_name(k)
            if name:
                all_names.add(name)
            for v in vs:
                name = normalize_actor_name(v)
                if name:
                    all_names.add(name)

    # Find all known names that appear as substrings in the response
    matches = [name for name in all_names if name in text_norm]

    if not matches:
        return ""  # No known actor found — treated as "X" (no prediction)

    # Return the longest match: "apt 41" is more specific than "apt"
    return max(matches, key=len)


def score_ctibench_taa(pred: str, gold: str, alias_dict: dict, related_dict: dict) -> dict:
    """Faithful CTI-TAA scoring using alias and related-group dictionaries."""
    pred_norm = normalize_actor_name(pred)
    gold_norm = normalize_actor_name(gold)

    connection = threat_actor_connection_ctibench(gold_norm, pred_norm, alias_dict, related_dict)

    correct = connection == "C"
    plausible = connection == "P"

    return {
        "pred": pred_norm,
        "gold": gold_norm,
        "valid": True,
        "connection": connection,
        "correct": correct,
        "plausible": plausible,
        "score": 1.0 if correct else 0.0,
        "combined_correct_or_plausible": correct or plausible,
    }


# ---------------------------------------------------------------------
# Collected-output evaluation helpers
# ---------------------------------------------------------------------

def load_collected_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stream_collected_jsonl(path: str):
    """Yield rows one at a time — use for large files to avoid OOM."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class _DetailWriter:
    """Context manager that writes detail JSONL rows one at a time."""
    def __init__(self, path: str):
        self.path = path
        self._f = None

    def __enter__(self):
        if self.path:
            out_dir = os.path.dirname(self.path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            self._f = open(self.path, "w", encoding="utf-8")
        return self

    def write(self, row: dict):
        if self._f is not None:
            self._f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def __exit__(self, *_):
        if self._f is not None:
            self._f.close()


def write_jsonl(path: str, rows: list):
    if not path:
        return

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_collected_pred_gold(row: dict) -> tuple:
    pred = row.get("model_response", "")
    gold = row.get("ground_truth", "")

    if not gold:
        gold = (
            row.get("metadata", {})
               .get("original_fields", {})
               .get("GT", "")
        )

    return pred, gold
# ---------------------------------------------------------------------
# Unified collected-output result wrapper
# ---------------------------------------------------------------------

def build_unified_result(
    benchmark: str,
    original_result: dict,
    input_jsonl: str,
    detailed_output: str = None,
) -> dict:
    benchmark = str(benchmark or "").lower()

    unified = {
        "benchmark": benchmark,
        "task": original_result.get("task"),
        "metric": original_result.get("metric"),
        "primary_score": None,
        "primary_score_name": None,
        "primary_score_unit": None,
        "correct": original_result.get("correct"),
        "total": (
            original_result.get("total")
            or original_result.get("total_rows")
            or original_result.get("valid_total")
            or original_result.get("success_count")
        ),
        "valid_total": (
            original_result.get("valid_total")
            or original_result.get("success_count")
            or original_result.get("total")
            or original_result.get("total_rows")
        ),
        "invalid": original_result.get("invalid", 0),
        "input_jsonl": input_jsonl,
        "detailed_output": detailed_output,
        "faithful_to": original_result.get("faithful_to"),
        "original_result": original_result,
    }

    # CTI-Bench
    if benchmark == "ctibench":
        if "accuracy_percent" in original_result:
            unified["primary_score"] = original_result["accuracy_percent"]
            unified["primary_score_name"] = "accuracy"
            unified["primary_score_unit"] = "percent"
            unified["metric"] = original_result.get("metric", "accuracy_percent")

        elif "mad" in original_result:
            unified["primary_score"] = original_result["mad"]
            unified["primary_score_name"] = "MAD"
            unified["primary_score_unit"] = "absolute_cvss_score_difference"
            unified["metric"] = "mean_absolute_deviation"

        elif "correct_plus_plausible_accuracy" in original_result:
            unified["primary_score"] = original_result["correct_plus_plausible_accuracy"]
            unified["primary_score_name"] = "correct_plus_plausible_accuracy"
            unified["primary_score_unit"] = "percent"
            unified["metric"] = "correct_and_plausible_accuracy"

    # AthenaBench
    elif benchmark == "athenabench":
        if "accuracy" in original_result:
            unified["primary_score"] = original_result["accuracy"]
            unified["primary_score_name"] = "accuracy"
            unified["primary_score_unit"] = "percent"
            unified["metric"] = "accuracy_percent"

        elif "f1" in original_result:
            unified["primary_score"] = original_result["f1"]
            unified["primary_score_name"] = "f1"
            unified["primary_score_unit"] = "percent"
            unified["metric"] = "f1_percent"

        elif "MAD" in original_result:
            unified["primary_score"] = original_result["MAD"]
            unified["primary_score_name"] = "MAD"
            unified["primary_score_unit"] = "absolute_cvss_score_difference"
            unified["metric"] = "mean_absolute_deviation"

    # CyberMetric
    elif benchmark == "cybermetric":
        unified["primary_score"] = original_result.get("accuracy")
        unified["primary_score_name"] = "accuracy"
        unified["primary_score_unit"] = "percent"
        unified["metric"] = "accuracy_percent"

    # SecEval
    elif benchmark == "seceval":
        score_float = original_result.get("score_float", {})
        unified["primary_score"] = score_float.get("Overall")
        unified["primary_score_name"] = "accuracy"
        unified["primary_score_unit"] = "percent"
        unified["metric"] = "accuracy_percent"

        score_fraction = original_result.get("score_fraction", {})
        if "Overall" in score_fraction:
            try:
                correct_str, total_str = score_fraction["Overall"].split("/")
                unified["correct"] = int(correct_str)
                unified["total"] = int(total_str)
            except Exception:
                pass

        unified["by_topic"] = {
            topic: {
                "accuracy_percent": value,
                "fraction": score_fraction.get(topic),
            }
            for topic, value in score_float.items()
            if topic != "Overall"
        }
    
    # RedSageMCQ
    elif benchmark == "redsage":
        unified["primary_score"] = original_result.get("regex_mcq_acc")
        unified["primary_score_name"] = "regex_mcq_acc"
        unified["primary_score_unit"] = "percent"
        unified["metric"] = "regex_mcq_acc"
        unified["correct"] = original_result.get("correct")
        unified["total"] = original_result.get("total")
        unified["valid_total"] = original_result.get("total")
        unified["invalid"] = original_result.get("invalid_extractions", 0)

    # SECURE
    elif benchmark == "secure":
        unified["primary_score"] = original_result.get("accuracy_percent")
        unified["primary_score_name"] = "accuracy"
        unified["primary_score_unit"] = "percent"
        unified["metric"] = "accuracy_percent"
        unified["correct"] = original_result.get("correct")
        unified["total"] = original_result.get("valid_total")
        unified["valid_total"] = original_result.get("valid_total")
        unified["invalid"] = original_result.get("invalid", 0)

    # CISSP
    elif benchmark == "cissp":
        unified["primary_score"] = original_result.get("accuracy_percent")
        unified["primary_score_name"] = "accuracy"
        unified["primary_score_unit"] = "percent"
        unified["metric"] = "accuracy_percent"

    # SecBench MCQ
    elif benchmark == "secbench":
        unified["primary_score"] = original_result.get("accuracy_percent")
        unified["primary_score_name"] = "accuracy"
        unified["primary_score_unit"] = "percent"
        unified["metric"] = "accuracy_percent"

    # MMLU-CS
    elif benchmark == "mmlu_cs":
        unified["primary_score"] = original_result.get("accuracy_percent")
        unified["primary_score_name"] = "accuracy"
        unified["primary_score_unit"] = "percent"
        unified["metric"] = "accuracy_percent"

    return unified

# ---------------------------------------------------------------------
# CTI-Bench collected-output evaluators
# Formatting from model-prediction.ipynb + scoring from evaluation.ipynb
# ---------------------------------------------------------------------

def evaluate_collected_ctibench_accuracy(
    input_jsonl_or_rows,
    scorer,
    formatter,
    detailed_output: str = None,
) -> dict:
    correct = 0
    total = 0
    invalid = 0
    total_rows = 0

    src = (stream_collected_jsonl(input_jsonl_or_rows)
           if isinstance(input_jsonl_or_rows, str)
           else iter(input_jsonl_or_rows))

    with _DetailWriter(detailed_output) as dw:
        for row in src:
            total_rows += 1
            raw_pred, gold = get_collected_pred_gold(row)
            formatted_pred = formatter(raw_pred)
            score_info = scorer(formatted_pred, gold)

            if score_info["valid"]:
                total += 1
            else:
                invalid += 1

            if score_info["correct"]:
                correct += 1

            dw.write({
                "task": row.get("task"),
                "index": row.get("index"),
                "formatted_pred": formatted_pred,
                "pred": score_info["pred"],
                "gold": score_info["gold"],
                "valid": score_info["valid"],
                "correct": score_info["correct"],
            })

    accuracy_percent = correct / total * 100 if total > 0 else 0.0

    return {
        "metric": "accuracy_percent",
        "accuracy_percent": accuracy_percent,
        "accuracy": accuracy_percent / 100,
        "correct": correct,
        "valid_total": total,
        "invalid": invalid,
        "total_rows": total_rows,
        "invalid_policy": "excluded_from_denominator",
        "faithful_to": "maveryn/cti-bench",
    }


def evaluate_collected_ctibench_rcm(rows: list, detailed_output: str = None) -> dict:
    correct = 0
    total = 0
    invalid = 0
    extraction_failures = 0
    detailed_rows = []

    for row in rows:
        raw_pred, gold = get_collected_pred_gold(row)
        formatted_pred, extraction_ok = ctibench_format_rcm(raw_pred)
        score_info = score_ctibench_rcm(formatted_pred, gold)

        if not extraction_ok:
            extraction_failures += 1

        if score_info["valid"]:
            total += 1
        else:
            invalid += 1
            print("Invalid response at row {}".format(int(row.get("index", -1)) + 1))

        if score_info["correct"]:
            correct += 1

        detailed_rows.append({
            "task": row.get("task"),
            "index": row.get("index"),
            "prompt": row.get("prompt"),
            "llm_output": raw_pred,
            "formatted_pred": formatted_pred,
            "extraction_ok": extraction_ok,
            "pred": score_info["pred"],
            "gold": score_info["gold"],
            "valid": score_info["valid"],
            "correct": score_info["correct"],
            "score_info": score_info,
        })

    accuracy_percent = correct / total * 100 if total > 0 else 0.0

    write_jsonl(detailed_output, detailed_rows)

    return {
        "metric": "accuracy_percent",
        "accuracy_percent": accuracy_percent,
        "accuracy": accuracy_percent / 100,
        "correct": correct,
        "valid_total": total,
        "invalid": invalid,
        "extraction_failures": extraction_failures,
        "total_rows": len(rows),
        "invalid_policy": "excluded_from_denominator",
        "faithful_to": "maveryn/cti-bench",
    }


def evaluate_collected_ctibench_vsp(
    rows: list,
    detailed_output: str = None,
    cvss_prefix: str = "CVSS:3.0/",
) -> dict:
    error_sum = 0.0
    total = 0
    invalid = 0
    extraction_failures = 0
    detailed_rows = []

    for row in rows:
        raw_pred, gold = get_collected_pred_gold(row)
        formatted_pred, extraction_ok = ctibench_format_vsp(raw_pred)
        score_info = score_ctibench_vsp(formatted_pred, gold, cvss_prefix=cvss_prefix)

        if not extraction_ok:
            extraction_failures += 1

        if score_info["valid"]:
            error_sum += score_info["error"]
            total += 1
        else:
            invalid += 1
            print("Invalid response at row {}".format(int(row.get("index", -1)) + 1))
            print(score_info.get("error_message"))

        detailed_rows.append({
            "task": row.get("task"),
            "index": row.get("index"),
            "prompt": row.get("prompt"),
            "llm_output": raw_pred,
            "formatted_pred": formatted_pred,
            "extraction_ok": extraction_ok,
            "pred": score_info["pred"],
            "gold": score_info["gold"],
            "valid": score_info["valid"],
            "pred_vector": score_info.get("pred_vector"),
            "pred_score": score_info.get("pred_score"),
            "gold_score": score_info.get("gold_score"),
            "error": score_info.get("error"),
            "error_message": score_info.get("error_message"),
            "score_info": score_info,
        })

    mad = error_sum / total if total > 0 else 0.0

    write_jsonl(detailed_output, detailed_rows)

    return {
        "metric": "mean_absolute_deviation",
        "mad": mad,
        "valid_total": total,
        "invalid": invalid,
        "extraction_failures": extraction_failures,
        "total_rows": len(rows),
        "cvss_prefix": cvss_prefix,
        "invalid_policy": "skipped",
        "faithful_to": "maveryn/cti-bench",
    }


def evaluate_collected_ctibench_taa(
    rows: list,
    alias_dict: dict,
    related_dict: dict,
    detailed_output: str = None,
    gt_tsv_path: str = None,
) -> dict:
    # Load ground-truth from TSV when JSONL rows have no ground_truth.
    # The official cti-taa.tsv has URL/Text/Prompt columns but no GT; the GT
    # is only in evaluation/responses/cti-taa-responses.tsv (GT column).
    # Rows align by index (0-based).
    gt_by_index = {}
    if gt_tsv_path and os.path.exists(gt_tsv_path):
        import csv as _csv
        with open(gt_tsv_path, newline="", encoding="utf-8") as f:
            for i, row_tsv in enumerate(_csv.DictReader(f, delimiter="\t")):
                gt_by_index[i] = row_tsv.get("GT", "")

    correct = 0
    plausible = 0
    total = 0
    detailed_rows = []

    for row in rows:
        raw_pred, gold = get_collected_pred_gold(row)
        # Fall back to TSV GT when ground_truth is absent (CTI-TAA dataset
        # does not embed GT in the public data file).
        if not gold and gt_by_index:
            gold = gt_by_index.get(int(row.get("index", -1)), "")

        # Extract a clean actor name from the verbose model response,
        # mirroring what CTI-Bench authors did manually in their TSV.
        extracted_pred = extract_actor_from_response(raw_pred, alias_dict, related_dict)

        pred = normalize_actor_name(extracted_pred)
        gt = normalize_actor_name(gold)

        res = threat_actor_connection_ctibench(gt, pred, alias_dict, related_dict)

        is_correct = res == "C"
        is_plausible = res == "P"

        if is_correct:
            correct += 1
        elif is_plausible:
            plausible += 1

        total += 1

        detailed_rows.append({
            "task": row.get("task"),
            "index": row.get("index"),
            "llm_output": raw_pred,
            "extracted_pred": extracted_pred,
            "pred": pred,
            "gold": gt,
            "connection": res,
            "correct": is_correct,
            "plausible": is_plausible,
            "combined_correct_or_plausible": is_correct or is_plausible,
        })

    correct_accuracy = correct / total * 100 if total > 0 else 0.0
    correct_plus_plausible_accuracy = (correct + plausible) / total * 100 if total > 0 else 0.0

    write_jsonl(detailed_output, detailed_rows)

    return {
        "metric": "correct_and_plausible_accuracy",
        "correct_accuracy": correct_accuracy,
        "correct_plus_plausible_accuracy": correct_plus_plausible_accuracy,
        "correct": correct,
        "plausible": plausible,
        "total": total,
        "faithful_to": "maveryn/cti-bench",
    }


def evaluate_collected_ctibench_ate(
    rows: list,
    detailed_output: str = None,
) -> dict:
    """CTI-ATE evaluation: accuracy via exact set match on MITRE T-IDs.

    GT is a comma-separated set of main technique IDs.
    Scoring follows the same approach as AthenaBench ATE:
      - sub-techniques stripped via .split(".")[0]
      - exact match (predicted set == gold set)
      - accuracy = correct / total

    Note: maveryn/cti-bench evaluation.ipynb does not implement ATE scoring.
    The paper (NeurIPS'24) states Micro-F1, but no reference implementation exists.
    """
    correct = 0
    total = len(rows)
    detailed_rows = []

    for row in rows:
        raw_pred, gold_str = get_collected_pred_gold(row)
        pred_ids = ctibench_format_ate(raw_pred)

        gold_ids = {
            t.split(".")[0].upper()
            for t in re.findall(r'\bT\d{4}(?:\.\d+)?\b', str(gold_str or ""), re.IGNORECASE)
        }

        is_correct = pred_ids == gold_ids
        if is_correct:
            correct += 1

        detailed_rows.append({
            "task": row.get("task"),
            "index": row.get("index"),
            "prompt": row.get("prompt"),
            "llm_output": raw_pred,
            "pred_ids": sorted(pred_ids),
            "gold_ids": sorted(gold_ids),
            "correct": is_correct,
        })

    accuracy_percent = correct / total * 100 if total > 0 else 0.0
    write_jsonl(detailed_output, detailed_rows)

    return {
        "metric": "accuracy_percent",
        "accuracy_percent": accuracy_percent,
        "accuracy": accuracy_percent / 100,
        "correct": correct,
        "total_rows": total,
        "faithful_to": "maveryn/cti-bench",
        "note": (
            "maveryn/cti-bench evaluation.ipynb does not implement ATE scoring. "
            "Uses exact set match consistent with AthenaBench ATE approach."
        ),
    }


def evaluate_collected_ctibench(
    input_jsonl: str,
    detailed_output: str = None,
    alias_dict_path: str = None,
    related_dict_path: str = None,
    cvss_prefix: str = "CVSS:3.0/",
    taa_gt_tsv_path: str = None,
) -> dict:
    rows = load_collected_jsonl(input_jsonl)

    if not rows:
        raise ValueError(f"No rows found in {input_jsonl}")

    task = rows[0].get("task", "").lower()

    if task in {"ctibench_mcq", "mcq"}:
        return evaluate_collected_ctibench_accuracy(
            rows,
            scorer=score_ctibench_mcq,
            formatter=ctibench_format_mcq,
            detailed_output=detailed_output,
        )

    if task in {"ctibench_rcm", "ctibench_rcm_2021", "rcm"}:
        return evaluate_collected_ctibench_rcm(
            rows,
            detailed_output=detailed_output,
        )

    if task in {"ctibench_vsp", "vsp"}:
        return evaluate_collected_ctibench_vsp(
            rows,
            detailed_output=detailed_output,
            cvss_prefix=cvss_prefix,
        )

    if task in {"ctibench_taa", "cti_taa", "taa"}:
        if not alias_dict_path or not related_dict_path:
            raise ValueError(
                "CTI-TAA requires --alias_dict_path and --related_dict_path."
            )

        import pickle

        with open(alias_dict_path, "rb") as f:
            alias_dict = pickle.load(f)

        with open(related_dict_path, "rb") as f:
            related_dict = pickle.load(f)

        return evaluate_collected_ctibench_taa(
            rows,
            alias_dict=alias_dict,
            related_dict=related_dict,
            detailed_output=detailed_output,
            gt_tsv_path=taa_gt_tsv_path,
        )

    if task in {"ctibench_ate", "ate"}:
        return evaluate_collected_ctibench_ate(
            rows,
            detailed_output=detailed_output,
        )

    raise ValueError(f"Unsupported CTI-Bench collected task: {task}")

# ---------------------------------------------------------------------
# AthenaBench faithful extraction helpers
# From AthenaBench athena_eval/answer_extractors.py
# ---------------------------------------------------------------------

ATHENA_PREFIX_RE = re.compile(
    r'^\s*(?:final\s+answer|answer|prediction|output|result)\s*[:\-–—]?\s*',
    re.IGNORECASE,
)


def athena_strip_prefix(s: str) -> str:
    return ATHENA_PREFIX_RE.sub("", s).strip()


def athena_extract_from_lines(text: str, pattern: str, transform=lambda x: x) -> str:
    lines = [ln.strip() for ln in str(text or "").strip().splitlines() if ln.strip()]

    for i in range(len(lines) - 1, -1, -1):
        raw = lines[i]
        line = athena_strip_prefix(raw)

        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return transform(match.group(1))

        if re.search(r"\banswer\b", raw, re.IGNORECASE):
            if i + 1 < len(lines):
                nxt = athena_strip_prefix(lines[i + 1])
                match = re.search(pattern, nxt, re.IGNORECASE)
                if match:
                    return transform(match.group(1))

            if i > 0:
                prv = athena_strip_prefix(lines[i - 1])
                match = re.search(pattern, prv, re.IGNORECASE)
                if match:
                    return transform(match.group(1))

    return ""


def athena_clean_freeform(s: str) -> str:
    s = s.strip()
    s = s.strip('"\'')
    s = re.sub(r"\s+", " ", s)
    return s


def athena_extract_answer(task: str, text: str) -> str:
    task = str(task or "").upper()

    if task == "RCM":
        return athena_extract_from_lines(text, r"(CWE-\d+)", lambda s: s.upper())

    if task == "VSP":
        return athena_extract_from_lines(text, r"(CVSS:3\.1/[^\s]+)", lambda s: s.strip())

    if task == "TAA":
        return athena_extract_from_lines(text, r"(.+)", athena_clean_freeform)

    if task == "RMS":
        line = athena_extract_from_lines(text, r"(.+)", athena_clean_freeform).upper()
        ids = re.findall(r"M\d{4}", line)
        return ", ".join(ids)

    if task == "ATE":
        tid = athena_extract_from_lines(
            text,
            r"(T\d{4}(?:\.\d{3})?)",
            lambda s: s.upper(),
        )
        return tid.split(".")[0]

    if task in {"CKT", "MCQ", "MCQ3K"}:
        return athena_extract_from_lines(text, r"\b([A-E])\b", lambda s: s.upper())

    return ""

# ---------------------------------------------------------------------
# AthenaBench faithful scoring helpers
# From AthenaBench athena_eval/evaluate.py
# ---------------------------------------------------------------------

def athena_load_alias_dict(path: str) -> dict:
    import csv

    alias = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row["ThreatActor"].strip().lower()
            v = row["Alias"].strip().lower()
            alias.setdefault(k, []).append(v)
            alias.setdefault(v, []).append(k)
    return alias


def athena_load_related_dict(path: str) -> dict:
    import csv

    related = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row["ThreatActor"].strip().lower()
            v = row["RelatedGroup"].strip().lower()
            related.setdefault(k, []).append(v)
            related.setdefault(v, []).append(k)
    return related


def athena_is_alias_connected(actor1: str, actor2: str, alias_dict: dict) -> bool:
    visited = set()
    queue = [actor1]

    while queue:
        cur = queue.pop(0)
        if cur == actor2:
            return True
        visited.add(cur)

        for nxt in alias_dict.get(cur, []):
            if nxt not in visited:
                queue.append(nxt)

    return False


def athena_is_related_connected(actor1: str, actor2: str, alias_dict: dict, related_dict: dict) -> bool:
    visited = set()
    queue = [actor1]

    while queue:
        cur = queue.pop(0)
        if cur == actor2:
            return True
        visited.add(cur)

        neighbours = alias_dict.get(cur, []) + related_dict.get(cur, [])
        for nxt in neighbours:
            if nxt not in visited:
                queue.append(nxt)

    return False


def athena_threat_actor_connection(actor1: str, actor2: str, alias_dict: dict, related_dict: dict) -> str:
    actor1 = actor1.strip().lower()
    actor2 = actor2.strip().lower()

    if athena_is_alias_connected(actor1, actor2, alias_dict):
        return "C"

    if athena_is_related_connected(actor1, actor2, alias_dict, related_dict):
        return "P"

    return "I"


def athena_score_taa(pred: str, ans: str, alias_dict: dict, related_dict: dict):
    res = athena_threat_actor_connection(ans, pred, alias_dict, related_dict)

    score = {
        "correct": 1 if res == "C" else 0,
        "plausible": 1 if res in {"C", "P"} else 0,
        "combined": 1.0 if res == "C" else 0.5 if res == "P" else 0.0,
    }
    return score, True


def athena_score_record(task: str, pred: str, ans: str, alias_dict=None, related_dict=None):
    task = str(task or "").upper()

    if task == "RCM":
        return (1 if pred.strip().lower() == ans.strip().lower() else 0, True)

    if task == "VSP":
        try:
            p = CVSS3(pred.strip()).scores()[0]
            a = CVSS3(ans.strip()).scores()[0]
            return (abs(p - a), True)
        except Exception:
            return (0.0, False)

    if task == "TAA":
        return athena_score_taa(pred, ans, alias_dict or {}, related_dict or {})

    if task == "ATE":
        p = pred.strip().split(".")[0].upper()
        a = ans.strip().split(".")[0].upper()
        return (1 if p and p == a else 0, True)

    if task == "RMS":
        p_ids = set(re.findall(r"M\d{4}", pred.upper()))
        a_ids = set(re.findall(r"M\d{4}", ans.upper()))

        tp = len(p_ids & a_ids)
        fp = len(p_ids - a_ids)
        fn = len(a_ids - p_ids)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        if precision == 0 and recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return (f1, True)

    return (1 if pred.strip().lower() == ans.strip().lower() else 0, True)


def infer_athena_task_name(row: dict) -> str:
    task = str(row.get("task", "") or "")

    mapping = {
        "athenabench_ckt": "CKT",
        "athenabench_ate": "ATE",
        "athenabench_rcm": "RCM",
        "athenabench_rms": "RMS",
        "athenabench_vsp": "VSP",
        "athenabench_taa": "TAA",
        "ckt": "CKT",
        "ate": "ATE",
        "rcm": "RCM",
        "rms": "RMS",
        "vsp": "VSP",
        "taa": "TAA",
    }

    if task.lower() in mapping:
        return mapping[task.lower()]

    meta_task = (
        row.get("metadata", {})
           .get("athena_eval_task")
    )
    if meta_task:
        return str(meta_task).upper()

    return task.upper()


def evaluate_collected_athenabench(
    input_jsonl: str,
    detailed_output: str = None,
    alias_csv_path: str = None,
    related_csv_path: str = None,
    vsp_denominator: float = 7.7,
) -> dict:
    rows = load_collected_jsonl(input_jsonl)

    if not rows:
        raise ValueError(f"No rows found in {input_jsonl}")

    task = infer_athena_task_name(rows[0])

    alias_dict = {}
    related_dict = {}

    if task == "TAA":
        if not alias_csv_path or not related_csv_path:
            raise ValueError(
                "AthenaBench TAA requires --athena_alias_csv_path and --athena_related_csv_path."
            )
        alias_dict = athena_load_alias_dict(alias_csv_path)
        related_dict = athena_load_related_dict(related_csv_path)

    scored_rows = []
    sum_score = 0.0
    count_success = 0

    sum_correct = 0
    sum_plausible = 0
    sum_combined = 0.0

    for row in rows:
        response = row.get("model_response", "")
        answer = row.get("ground_truth", "")

        if not answer:
            answer = (
                row.get("metadata", {})
                   .get("original_fields", {})
                   .get("answer", "")
            )

        pred = athena_extract_answer(task, response)
        score, success = athena_score_record(
            task,
            pred,
            answer,
            alias_dict=alias_dict,
            related_dict=related_dict,
        )

        out_row = {
            **row,
            "athena_task": task,
            "response": response,
            "answer": answer,
            "prediction": pred,
            "score": score,
            "success": success,
        }
        scored_rows.append(out_row)

        if success:
            count_success += 1

            if isinstance(score, dict):
                sum_correct += score.get("correct", 0)
                sum_plausible += score.get("plausible", 0)
                sum_combined += score.get("combined", 0.0)
            else:
                sum_score += float(score)

    write_jsonl(detailed_output, scored_rows)

    if task == "TAA":
        return {
            "benchmark": "AthenaBench",
            "task": task,
            "accuracy": (sum_correct / count_success * 100) if count_success else 0.0,
            "plausible_accuracy": (sum_plausible / count_success * 100) if count_success else 0.0,
            "combined_accuracy": (sum_combined / count_success * 100) if count_success else 0.0,
            "success_count": count_success,
            "total_rows": len(rows),
            "faithful_to": "AthenaBench athena_eval/evaluate.py",
        }

    if task == "RMS":
        return {
            "benchmark": "AthenaBench",
            "task": task,
            "f1": (sum_score / count_success * 100) if count_success else 0.0,
            "success_count": count_success,
            "total_rows": len(rows),
            "faithful_to": "AthenaBench athena_eval/evaluate.py",
        }

    if task == "VSP":
        mad = sum_score / count_success if count_success else 0.0
        denom = vsp_denominator if vsp_denominator else 1.0
        accuracy = 1 - (mad / denom)

        return {
            "benchmark": "AthenaBench",
            "task": task,
            "MAD": mad,
            "accuracy": accuracy * 100,
            "success_count": count_success,
            "total_rows": len(rows),
            "vsp_denominator": denom,
            "faithful_to": "AthenaBench athena_eval/evaluate.py",
        }

    accuracy = sum_score / count_success if count_success else 0.0

    return {
        "benchmark": "AthenaBench",
        "task": task,
        "accuracy": accuracy * 100,
        "success_count": count_success,
        "total_rows": len(rows),
        "faithful_to": "AthenaBench athena_eval/evaluate.py",
    }


# ---------------------------------------------------------------------
# CyberMetric faithful collected-output evaluator
# From cybermetric/CyberMetric CyberMetric_evaluator.py
# ---------------------------------------------------------------------

def cybermetric_extract_answer(response: str):
    response = str(response or "")

    if response.strip():
        match = re.search(r"ANSWER:?\s*([A-D])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def evaluate_collected_cybermetric(
    input_jsonl: str,
    detailed_output: str = None,
) -> dict:
    rows = load_collected_jsonl(input_jsonl)

    if not rows:
        raise ValueError(f"No rows found in {input_jsonl}")

    correct_count = 0
    incorrect_answers = []
    detailed_rows = []

    for row in rows:
        response = row.get("model_response", "")
        correct_answer = row.get("ground_truth", "")

        if not correct_answer:
            correct_answer = (
                row.get("metadata", {})
                   .get("original_fields", {})
                   .get("solution", "")
            )

        correct_answer = str(correct_answer or "").strip().upper()
        llm_answer = cybermetric_extract_answer(response)

        is_correct = llm_answer == correct_answer

        if is_correct:
            correct_count += 1
        else:
            incorrect_answers.append({
                "index": row.get("index"),
                "question": row.get("prompt"),
                "correct_answer": correct_answer,
                "llm_answer": llm_answer,
            })

        detailed_rows.append({
            "task": row.get("task"),
            "index": row.get("index"),
            "prompt": row.get("prompt"),
            "llm_output": response,
            "pred": llm_answer,
            "gold": correct_answer,
            "correct": is_correct,
        })

    write_jsonl(detailed_output, detailed_rows)

    total = len(rows)
    accuracy = correct_count / total * 100 if total else 0.0

    return {
        "benchmark": "CyberMetric",
        "metric": "accuracy_percent",
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total,
        "incorrect": len(incorrect_answers),
        "faithful_to": "cybermetric/CyberMetric CyberMetric_evaluator.py",
        "extraction": r"ANSWER:?\s*([A-D])",
        "fallback_extraction": False,
    }

# ---------------------------------------------------------------------
# SecEval faithful collected-output evaluator
# From XuanwuAI/SecEval eval/eval.py
# ---------------------------------------------------------------------

def seceval_extract_answer(response: str) -> tuple[str, str]:
    """
    Faithful to SecEval eval.py:
    - if "Answer:" in llm_output, remove it
    - llm_answer = sorted unique A-D letters found in llm_output
    """
    llm_output = str(response or "")

    if "Answer:" in llm_output:
        llm_output = llm_output.replace("Answer:", "")

    llm_answer = "".join(
        sorted(list(set(re.findall(r"[A-D]", llm_output))))
    )

    return llm_output, llm_answer


def seceval_count_score_by_topic(evaluated_rows: list) -> tuple[dict, dict]:
    score_by_topic = {}
    total_score_by_topic = {}
    score = 0

    for row in evaluated_rows:
        topics = row.get("topics", ["Unknown"])

        if isinstance(topics, str):
            topics = [topics]

        for topic in topics:
            if topic not in score_by_topic:
                score_by_topic[topic] = 0
                total_score_by_topic[topic] = 0

            score_by_topic[topic] += row["score"]
            total_score_by_topic[topic] += 1

        score += row["score"]

    score_fraction = {
        k: f"{v}/{total_score_by_topic[k]}"
        for k, v in score_by_topic.items()
    }

    score_float = {
        k: round(100 * float(v) / float(total_score_by_topic[k]), 4)
        for k, v in score_by_topic.items()
    }

    score_float["Overall"] = (
        round(100 * float(score) / float(len(evaluated_rows)), 4)
        if evaluated_rows else 0.0
    )
    score_fraction["Overall"] = f"{score}/{len(evaluated_rows)}"

    return score_fraction, score_float


def evaluate_collected_seceval(
    input_jsonl: str,
    detailed_output: str = None,
) -> dict:
    rows = load_collected_jsonl(input_jsonl)

    if not rows:
        raise ValueError(f"No rows found in {input_jsonl}")

    evaluated_rows = []

    for row in rows:
        response = row.get("model_response", "")
        gold = row.get("ground_truth", "")

        original_fields = (
            row.get("metadata", {})
               .get("original_fields", {})
        )

        if not gold:
            gold = original_fields.get("answer", "")

        topics = (
            original_fields.get("topics")
            or row.get("metadata", {}).get("topics")
            or ["Unknown"]
        )

        if isinstance(topics, str):
            topics = [topics]

        llm_output, llm_answer = seceval_extract_answer(response)

        # Faithful to SecEval eval.py: normalize correct_answer with sorted(upper())
        gold_norm = "".join(sorted(str(gold or "").upper()))
        score = int(llm_answer == gold_norm)

        evaluated_row = {
            **original_fields,
            "task": row.get("task"),
            "index": row.get("index"),
            "prompt": row.get("prompt"),
            "topics": topics,
            "answer": gold_norm,
            "llm_output": llm_output,
            "llm_answer": llm_answer,
            "score": score,
        }

        evaluated_rows.append(evaluated_row)

    score_fraction, score_float = seceval_count_score_by_topic(evaluated_rows)

    result_with_score = {
        "score_fraction": score_fraction,
        "score_float": score_float,
        "benchmark": "SecEval",
        "faithful_to": "XuanwuAI/SecEval eval/eval.py",
        "extraction": 'remove "Answer:" then sorted unique re.findall(r"[A-D]")',
    }

    write_jsonl(detailed_output, evaluated_rows)

    return result_with_score

# ---------------------------------------------------------------------
# RedSageMCQ faithful collected-output evaluator
# From RISys-Lab/RedSage eval/cybersecurity_benchmarks.py
# redsage_mcq_em:{subset}: exact_match, prefix_exact_match, regex_mcq_acc
# ---------------------------------------------------------------------

REDSAGE_MCQ_LETTERS = ["A", "B", "C", "D"]


def redsage_extract_mcq_answer(text: str) -> str:
    """Faithful to RedSage MCQAcc extraction for generative MCQ outputs."""
    if not text:
        return ""

    text = str(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    pattern_standalone = re.compile(r"\b([ABCD])\b")
    pattern_leading = re.compile(r"^([ABCD])[).:]")

    def find_in_line(line: str):
        cleaned = line.replace("**", " ")

        m = pattern_leading.match(cleaned)
        if m:
            return m.group(1)

        m = pattern_standalone.search(cleaned)
        if m:
            return m.group(1)

        # Trailing markdown emphasis like **A**
        m = re.search(r"\*\*([ABCD])\*\*$", cleaned)
        if m:
            return m.group(1)

        # Ending with letter where previous character is not alphanumeric.
        if cleaned and cleaned[-1] in "ABCD" and cleaned[-2:-1].isalnum() is False:
            return cleaned[-1]

        return None

    answer = None

    if lines:
        for line in lines[::-1][:5]:
            answer = find_in_line(line)
            if answer:
                break

    if not answer:
        letters = re.findall(r"[ABCD]", text)
        unique = set(letters)
        if len(unique) == 1:
            answer = letters[0]

    return answer.upper() if answer else ""


def score_redsage_generative(pred: str, gold: str) -> dict:
    """Score RedSage redsage_mcq_em generated output.

    RedSage _em tasks report:
    - exact_match
    - prefix_exact_match
    - regex_mcq_acc

    The regex_mcq_acc uses MCQAcc extraction and counts extraction failure as 0.
    """
    pred_raw = str(pred or "").strip()
    pred_upper = pred_raw.upper()
    gold_norm = str(gold or "").strip().upper()

    extracted = redsage_extract_mcq_answer(pred_raw)

    exact_match = bool(gold_norm) and pred_upper == gold_norm
    prefix_exact_match = bool(gold_norm) and pred_upper.startswith(gold_norm)
    regex_mcq_acc = bool(extracted) and extracted == gold_norm

    return {
        "pred": extracted,
        "gold": gold_norm,
        "valid": bool(extracted),
        "correct": regex_mcq_acc,
        "score": 1.0 if regex_mcq_acc else 0.0,
        "exact_match": exact_match,
        "prefix_exact_match": prefix_exact_match,
        "regex_mcq_acc": regex_mcq_acc,
    }


def evaluate_collected_redsage(
    input_jsonl: str,
    detailed_output: str = None,
) -> dict:
    exact_match_total = 0
    prefix_exact_match_total = 0
    regex_mcq_total = 0
    valid_extractions = 0
    total = 0

    with _DetailWriter(detailed_output) as writer:
        for row in stream_collected_jsonl(input_jsonl):
            total += 1
            response = row.get("model_response", "")
            gold = row.get("ground_truth", "")

            if not gold:
                gold = (
                    row.get("metadata", {})
                       .get("original_fields", {})
                       .get("solution", "")
                )

            score_info = score_redsage_generative(response, gold)

            if score_info["valid"]:
                valid_extractions += 1
            if score_info["exact_match"]:
                exact_match_total += 1
            if score_info["prefix_exact_match"]:
                prefix_exact_match_total += 1
            if score_info["regex_mcq_acc"]:
                regex_mcq_total += 1

            writer.write({
                "task": row.get("task"),
                "index": row.get("index"),
                "pred": score_info["pred"],
                "gold": score_info["gold"],
                "valid": score_info["valid"],
                "correct": score_info["correct"],
                "score_info": score_info,
            })

    if not total:
        raise ValueError(f"No rows found in {input_jsonl}")

    return {
        "benchmark": "RedSageMCQ",
        "metric": "regex_mcq_acc",
        "accuracy": (regex_mcq_total / total * 100) if total else 0.0,
        "correct": regex_mcq_total,
        "total": total,
        "valid_extractions": valid_extractions,
        "invalid_extractions": total - valid_extractions,
        "exact_match": (exact_match_total / total * 100) if total else 0.0,
        "prefix_exact_match": (prefix_exact_match_total / total * 100) if total else 0.0,
        "regex_mcq_acc": (regex_mcq_total / total * 100) if total else 0.0,
        "faithful_to": (
            "RISys-Lab/RedSage redsage_mcq_em task: "
            "exact_match, prefix_exact_match, regex_mcq_acc"
        ),
        "note": (
            "This evaluates RedSage redsage_mcq_em:{subset}-style generative outputs. "
            "It uses include_context=False, generation_size=100, stop_sequence=['\\n'], "
            "and reports exact_match, prefix_exact_match, and regex_mcq_acc. "
            "It does not reproduce the default redsage_mcq:{subset} loglikelihood_acc mode."
        ),
    }

# ---------------------------------------------------------------------
# SECURE faithful collected-output evaluators
# aiforsec/SECURE: MAET, CWET (MCQ A/B/C/D/X) and KCV (T/F/X)
# No public eval script; faithful to prompt wording and paper metric.
# ---------------------------------------------------------------------

SECURE_MCQ_VALID = {"A", "B", "C", "D", "X"}
SECURE_TF_VALID = {"T", "F", "X"}


def secure_extract_mcq_answer(text: str) -> str:
    """Extract A/B/C/D/X from a SECURE MAET/CWET response.

    Prompt asks for exactly one letter. Try:
    1. Stripped full text is the letter.
    2. Last non-empty line is or contains the letter (standalone).
    3. Last standalone letter in entire response.
    """
    text = str(text or "").strip()
    if text.upper() in SECURE_MCQ_VALID:
        return text.upper()
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.upper() in SECURE_MCQ_VALID:
            return line.upper()
        m = re.search(r'(?<![A-Za-z])([ABCDX])(?![A-Za-z])', line.upper())
        if m:
            return m.group(1)
    matches = re.findall(r'(?<![A-Za-z])([ABCDX])(?![A-Za-z])', text.upper())
    return matches[-1] if matches else ""


_TF_WORD_MAP = {"TRUE": "T", "FALSE": "F"}


def secure_extract_tf_answer(text: str) -> str:
    """Extract T/F/X from a SECURE KCV response.

    Handles bare letter (T/F/X), word forms (True/False), and fallback search.
    """
    text = str(text or "").strip()
    if text.upper() in SECURE_TF_VALID:
        return text.upper()
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.upper() in SECURE_TF_VALID:
            return line.upper()
        # Word forms: "True." / "False" / "true" etc.
        for word, letter in _TF_WORD_MAP.items():
            if re.search(r'\b' + word + r'\b', line, re.IGNORECASE):
                return letter
        # Standalone letter T/F/X (not part of a longer word)
        m = re.search(r'(?<![A-Za-z])([TFX])(?![A-Za-z])', line.upper())
        if m:
            return m.group(1)
    # Full-text fallback: word forms first, then standalone letter
    for word, letter in _TF_WORD_MAP.items():
        if re.search(r'\b' + word + r'\b', text, re.IGNORECASE):
            return letter
    matches = re.findall(r'(?<![A-Za-z])([TFX])(?![A-Za-z])', text.upper())
    return matches[-1] if matches else ""


def evaluate_collected_secure(
    input_jsonl: str,
    detailed_output: str = None,
) -> dict:
    rows = load_collected_jsonl(input_jsonl)

    if not rows:
        raise ValueError(f"No rows found in {input_jsonl}")

    task = rows[0].get("task", "").lower()

    if task == "secure_kcv":
        extractor = secure_extract_tf_answer
        valid_set = SECURE_TF_VALID
    else:
        extractor = secure_extract_mcq_answer
        valid_set = SECURE_MCQ_VALID

    correct = 0
    total = 0
    invalid = 0
    detailed_rows = []

    for row in rows:
        raw_pred = row.get("model_response", "")
        gold = row.get("ground_truth", "")
        if not gold:
            gold = (
                row.get("metadata", {})
                   .get("original_fields", {})
                   .get("Correct Answer", "")
            )
        gold = str(gold or "").strip().upper()

        pred = extractor(raw_pred)
        valid = pred in valid_set
        correct_flag = valid and pred == gold

        if valid:
            total += 1
        else:
            invalid += 1

        if correct_flag:
            correct += 1

        detailed_rows.append({
            "task": row.get("task"),
            "index": row.get("index"),
            "prompt": row.get("prompt"),
            "llm_output": raw_pred,
            "pred": pred,
            "gold": gold,
            "valid": valid,
            "correct": correct_flag,
        })

    accuracy_percent = correct / total * 100 if total > 0 else 0.0
    write_jsonl(detailed_output, detailed_rows)

    return {
        "benchmark": "SECURE",
        "task": task,
        "metric": "accuracy_percent",
        "accuracy_percent": accuracy_percent,
        "accuracy": accuracy_percent / 100,
        "correct": correct,
        "valid_total": total,
        "invalid": invalid,
        "total_rows": len(rows),
        "invalid_policy": "excluded_from_denominator",
        "faithful_to": "aiforsec/SECURE",
    }


# ---------------------------------------------------------------------
# CISSP collected-output evaluator
# No official eval script; single-choice MCQ exact match.
# ---------------------------------------------------------------------

def cissp_extract_mcq_answer(text: str) -> str:
    """Extract A/B/C/D from a CISSP response."""
    text = str(text or "").strip()
    if text.upper() in {"A", "B", "C", "D"}:
        return text.upper()
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.upper() in {"A", "B", "C", "D"}:
            return line.upper()
        m = re.search(r'(?<![A-Za-z])([ABCD])(?![A-Za-z])', line.upper())
        if m:
            return m.group(1)
    matches = re.findall(r'(?<![A-Za-z])([ABCD])(?![A-Za-z])', text.upper())
    return matches[-1] if matches else ""


def evaluate_collected_cissp(
    input_jsonl: str,
    detailed_output: str = None,
) -> dict:
    rows = load_collected_jsonl(input_jsonl)

    if not rows:
        raise ValueError(f"No rows found in {input_jsonl}")

    correct = 0
    total = len(rows)
    detailed_rows = []

    for row in rows:
        response = row.get("model_response", "")
        gold = str(row.get("ground_truth", "") or "").strip().upper()

        pred = cissp_extract_mcq_answer(response)
        is_correct = bool(pred) and pred == gold

        if is_correct:
            correct += 1

        detailed_rows.append({
            "task": row.get("task"),
            "index": row.get("index"),
            "prompt": row.get("prompt"),
            "llm_output": response,
            "pred": pred,
            "gold": gold,
            "correct": is_correct,
        })

    accuracy = correct / total * 100 if total else 0.0
    write_jsonl(detailed_output, detailed_rows)

    return {
        "benchmark": "CISSP",
        "metric": "accuracy_percent",
        "accuracy_percent": accuracy,
        "accuracy": accuracy / 100,
        "correct": correct,
        "total": total,
        "faithful_to": "local-CISSP-dataset",
    }


# ---------------------------------------------------------------------
# SecBench MCQ collected-output evaluator
# No official eval script; exact-match of sorted letter(s).
# ---------------------------------------------------------------------

def secbench_extract_mcq_answer(text: str) -> str:
    """Extract and normalize letter(s) A-D from a SecBench MCQ response.

    SecBench labels can be single ("A") or multi-answer ("AB", "BCD").
    The inference prompt asks for only the letter(s). Extract all A-D letters
    found in the response, deduplicate and sort to match the label format.
    """
    text = str(text or "").strip()
    letters = re.findall(r'(?<![A-Za-z])([ABCD])(?![A-Za-z])', text.upper())
    if not letters:
        letters = re.findall(r'[ABCD]', text.upper())
    return "".join(sorted(set(letters)))


def evaluate_collected_secbench_mcq(
    input_jsonl: str,
    detailed_output: str = None,
) -> dict:
    rows = load_collected_jsonl(input_jsonl)

    if not rows:
        raise ValueError(f"No rows found in {input_jsonl}")

    correct = 0
    total = len(rows)
    detailed_rows = []

    for row in rows:
        response = row.get("model_response", "")
        gold = str(row.get("ground_truth", "") or "").strip()
        gold_norm = "".join(sorted(set(re.findall(r'[ABCD]', gold.upper()))))

        pred = secbench_extract_mcq_answer(response)
        is_correct = bool(pred) and pred == gold_norm

        if is_correct:
            correct += 1

        detailed_rows.append({
            "task": row.get("task"),
            "index": row.get("index"),
            "prompt": row.get("prompt"),
            "llm_output": response,
            "pred": pred,
            "gold": gold_norm,
            "correct": is_correct,
        })

    accuracy = correct / total * 100 if total else 0.0
    write_jsonl(detailed_output, detailed_rows)

    return {
        "benchmark": "SecBench",
        "metric": "accuracy_percent",
        "accuracy_percent": accuracy,
        "accuracy": accuracy / 100,
        "correct": correct,
        "total": total,
        "faithful_to": "secbench-git/SecBench",
        "extraction_note": (
            "SecBench paper/repo provide no eval script. "
            "Exact match of sorted unique A-D letter(s) between prediction and label. "
            "No partial credit for multi-answer questions."
        ),
    }


# ---------------------------------------------------------------------
# MMLU-CS collected-output evaluator
# Faithful to hendrycks/test evaluate.py letter-exact-match.
# Supports both generation (text response) and logprobs (letter pre-scored) modes.
# ---------------------------------------------------------------------

def mmlu_extract_answer(text: str) -> str:
    """Extract A/B/C/D from an MMLU-CS MCQ generation response."""
    text = str(text or "").strip()
    if text.upper() in {"A", "B", "C", "D"}:
        return text.upper()
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.upper() in {"A", "B", "C", "D"}:
            return line.upper()
        m = re.search(r'(?<![A-Za-z])([ABCD])(?![A-Za-z])', line.upper())
        if m:
            return m.group(1)
    matches = re.findall(r'(?<![A-Za-z])([ABCD])(?![A-Za-z])', text.upper())
    return matches[-1] if matches else ""


def evaluate_collected_mmlu_cs(
    input_jsonl: str,
    detailed_output: str = None,
) -> dict:
    rows = load_collected_jsonl(input_jsonl)

    if not rows:
        raise ValueError(f"No rows found in {input_jsonl}")

    correct = 0
    total = len(rows)
    detailed_rows = []

    for row in rows:
        response = row.get("model_response", "")
        gold = str(row.get("ground_truth", "") or "").strip().upper()
        task_name = row.get("task", "")

        # For logprobs mode model_response is already the argmax prediction letter.
        if task_name in {"mmlu-cs-logprobs", "mmlu_cs_logprobs"}:
            pred = str(response or "").strip().upper()
        else:
            pred = mmlu_extract_answer(response)

        is_correct = bool(pred) and pred == gold

        if is_correct:
            correct += 1

        detailed_rows.append({
            "task": task_name,
            "index": row.get("index"),
            "llm_output": response,
            "pred": pred,
            "gold": gold,
            "correct": is_correct,
        })

    accuracy = correct / total * 100 if total else 0.0
    write_jsonl(detailed_output, detailed_rows)

    return {
        "benchmark": "MMLU-CS",
        "metric": "accuracy_percent",
        "accuracy_percent": accuracy,
        "accuracy": accuracy / 100,
        "correct": correct,
        "total": total,
        "faithful_to": "hendrycks/test evaluate.py",
        "extraction_note": (
            "Official MMLU scoring selects the highest-logprob answer among "
            "' A', ' B', ' C', ' D'. For the generation mode this evaluator "
            "uses last standalone A/B/C/D letter in the response."
        ),
    }


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_path: str, base_model: str = None, is_base: bool = False):
    _ensure_inference_deps()
    """Load model (base or fine-tuned) with optional LoRA adapters
    
    Args:
        model_path: Path to model checkpoint
        base_model: Base model name (required if model_path contains LoRA adapters)
        is_base: If True, load as base model without any adapters
    """
    if is_base:
        print(f"Loading BASE model from: {model_path}")
        model_path_to_load = model_path
    else:
        print(f"Loading FINE-TUNED model from: {model_path}")
        model_path_to_load = model_path
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model if (base_model and not is_base) else model_path_to_load, 
        trust_remote_code=True
    )
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if is_base:
        # Load base model only (pre-training evaluation)
        model = AutoModelForCausalLM.from_pretrained(
            model_path_to_load,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    elif os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # LoRA checkpoint (post-training evaluation)
        print("Loading LoRA adapters...")
        if not base_model:
            raise ValueError("base_model must be specified when loading LoRA adapters")
        if not HAS_PEFT:
            raise ImportError(
                "PEFT library not found but LoRA adapter detected. "
                "Install peft with: pip install peft"
            )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        # Full merged model
        model = AutoModelForCausalLM.from_pretrained(
            model_path_to_load,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    return model, tokenizer


def chat_completion_api(endpoint: str, model_name: str, prompt: str, 
                       api_key: str = "", max_tokens: int = 1024, 
                       temperature: float = 0.1, retries: int = 3, 
                       api_version: str = "2024-02-15-preview") -> str:
    """Call OpenAI-compatible API endpoint (supports both OpenAI and Azure OpenAI)"""
    
    # Detect if this is Azure endpoint
    is_azure = "cognitiveservices.azure.com" in endpoint or "openai.azure.com" in endpoint
    
    headers = {"Content-Type": "application/json"}
    
    if is_azure:
        # Azure OpenAI authentication
        headers["api-key"] = api_key
        # For Azure, model_name is the deployment name
        # Format: {endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={version}
        if endpoint.endswith("/"):
            endpoint = endpoint.rstrip("/")
        url = f"{endpoint}/openai/deployments/{model_name}/chat/completions?api-version={api_version}"
    else:
        # Standard OpenAI API
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = endpoint
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    # Only add model to payload for non-Azure (Azure uses deployment in URL)
    if not is_azure:
        payload["model"] = model_name
    
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            if r.status_code != 200:
                print(f"API returned status {r.status_code}: {r.text[:200]}")
                time.sleep(2 ** attempt)
                continue
            data = r.json()
            if not data.get("choices"):
                print(f"API response missing 'choices': {data}")
                time.sleep(2 ** attempt)
                continue
            
            choice = data["choices"][0]
            msg = choice.get("message", {}) or {}
            content = msg.get("content") or msg.get("reasoning_content") or choice.get("text") or ""
            return str(content).strip()
        except Exception as e:
            print(f"API call failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    
    return ""


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 1024,
                     use_api: bool = False, api_endpoint: str = None,
                     api_model: str = None, api_key: str = "", api_version: str = None):
    """Generate response from model (local or API)"""
    _ensure_inference_deps()
    
    # API mode
    if use_api:
        if not api_endpoint or not api_model:
            raise ValueError("API endpoint and model name required for API mode")
        
        # Get API version from parameter or environment (for Azure)
        if api_version is None:
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        return chat_completion_api(api_endpoint, api_model, prompt, api_key, max_new_tokens, 
                                   api_version=api_version)
    
    # Local model mode
    # Always try to use tokenizer's native chat template first
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        # Fallback to model-specific templates
        model_name = tokenizer.name_or_path.lower()
        
        if "gemma" in model_name:
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif "llama" in model_name:
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def extract_final_answer(text: str, task_type: str = None) -> str:
    """Extract final answer from model response
    
    For MCQ: Looks for A/B/C/D letters, preferring explicit answer patterns
    For other tasks: Returns last non-empty line
    """
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    
    if task_type == "mcq":
        # For MCQ, look for A, B, C, or D with specific priority
        
        # Priority 1: "Answer: X" or "Final Answer: X" patterns (most reliable)
        answer_patterns = [
            r'\*\*(?:Final )?Answer:\*\*\s*([A-D])\b',  # **Answer:** B or **Final Answer:** B
            r'(?:Final )?Answer:\s*([A-D])\b',           # Answer: B or Final Answer: B
        ]
        for pattern in answer_patterns:
            for line in reversed(lines):
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1).upper()
        
        # Priority 2: Single letter lines (exact match)
        for line in reversed(lines):
            if line.upper() in ['A', 'B', 'C', 'D']:
                return line.upper()
        
        # Priority 3: Lines starting with the letter (not followed by ")")
        for line in reversed(lines):
            first_char = line[0].upper() if line else ''
            if first_char in ['A', 'B', 'C', 'D'] and (len(line) == 1 or line[1] != ')'):
                return first_char
        
        # Priority 4: Any occurrence of A/B/C/D in reverse order
        mcq_pattern = r'\b([A-D])\b'
        for line in reversed(lines):
            match = re.search(mcq_pattern, line.upper())
            if match:
                return match.group(1)
    
    # Default: return last line
    return lines[-1] if lines else text


def parse_ids_from_text(text: str) -> set:
    """Extract MITRE technique IDs or CWE IDs from text"""
    # Pattern for MITRE IDs: T1234 or T1234.567
    # Pattern for CWE IDs: CWE-123
    pattern = r'\b(?:T\d{4}(?:\.\d{3})?|CWE-\d+)\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return set([m.upper() for m in matches])


def compute_set_metrics(pred_set: set, gold_set: set) -> dict:
    """Compute precision, recall, F1 for set-based predictions"""
    if not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": len(pred_set) == 0}
    
    true_positives = len(pred_set & gold_set)
    
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(gold_set) if gold_set else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = pred_set == gold_set
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match
    }


def evaluate_mcq(model, tokenizer, dataset, detailed_output=None, **api_kwargs):
    """Evaluate CTI-Bench MCQ using original validity/exact-match logic."""
    print("\n" + "="*50)
    print("Evaluating CTI-MCQ")
    print("="*50)

    correct = 0
    total = 0
    invalid = 0
    detailed_results = []

    for idx, sample in enumerate(tqdm(dataset, desc="CTI-MCQ")):
        prompt = sample["Prompt"]
        ground_truth = sample.get("GT", "").strip()

        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        score_info = score_ctibench_mcq(response, ground_truth)

        if score_info["valid"]:
            total += 1
            if score_info["correct"]:
                correct += 1
        else:
            invalid += 1
            print(f"Invalid response at row {idx + 1}")

        if detailed_output is not None:
            detailed_results.append({
                "index": idx,
                "prompt": prompt,
                "llm_output": response,
                "pred": score_info["pred"],
                "gold": score_info["gold"],
                "valid": score_info["valid"],
                "correct": score_info["correct"],
                "score_info": score_info,
            })

    accuracy = correct / total if total > 0 else 0.0

    print(f"\nAccuracy: {accuracy * 100:.4f} ({correct}/{total} valid)")
    print(f"Invalid responses: {invalid}")

    if detailed_output is not None:
        with open(detailed_output, "w", encoding="utf-8") as f:
            for result in detailed_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Detailed results saved to: {detailed_output}")

    return {
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "correct": correct,
        "valid_total": total,
        "invalid": invalid,
    }


def evaluate_rcm(model, tokenizer, dataset, detailed_output=None, **api_kwargs):
    """Evaluate CTI-Bench RCM/RCM-2021 using original startswith/exact-match logic."""
    print("\n" + "="*50)
    print("Evaluating CTI-RCM")
    print("="*50)

    correct = 0
    total = 0
    invalid = 0
    detailed_results = []

    for idx, sample in enumerate(tqdm(dataset, desc="CTI-RCM")):
        prompt = sample["Prompt"]
        ground_truth = sample.get("GT", "").strip()

        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        score_info = score_ctibench_rcm(response, ground_truth)

        if score_info["valid"]:
            total += 1
            if score_info["correct"]:
                correct += 1
        else:
            invalid += 1
            print(f"Invalid response at row {idx + 1}")

        if detailed_output is not None:
            detailed_results.append({
                "index": idx,
                "prompt": prompt,
                "llm_output": response,
                "pred": score_info["pred"],
                "gold": score_info["gold"],
                "valid": score_info["valid"],
                "correct": score_info["correct"],
                "score_info": score_info,
            })

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy * 100:.4f} ({correct}/{total} valid)")
    print(f"Invalid responses: {invalid}")

    if detailed_output is not None:
        with open(detailed_output, "w", encoding="utf-8") as f:
            for result in detailed_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Detailed results saved to: {detailed_output}")

    return {
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "correct": correct,
        "valid_total": total,
        "invalid": invalid,
    }


def evaluate_vsp(model, tokenizer, dataset, detailed_output=None, **api_kwargs):
    """Evaluate on CTI-VSP (CVSS score prediction)"""
    print("\n" + "="*50)
    print("Evaluating CTI-VSP (CVSS Score Prediction)")
    print("="*50)
    
    errors = []
    detailed_results = []
    
    for idx, sample in enumerate(tqdm(dataset, desc="CTI-VSP")):
        prompt = sample['Prompt']
        ground_truth = sample.get('GT', '').strip()
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        answer = extract_final_answer(response)
        
        # Extract numeric CVSS score
        try:
            # Try to find a number in ground truth
            gold_score = float(re.search(r'\d+\.?\d*', ground_truth).group())
            pred_score = float(re.search(r'\d+\.?\d*', answer).group())
            
            error = abs(pred_score - gold_score)
            errors.append(error)
            
            # Save detailed result
            if detailed_output is not None:
                detailed_results.append({
                    "index": idx,
                    "prompt": prompt,
                    "llm_output": response,
                    "pred": pred_score,
                    "gold": gold_score,
                    "error": error
                })
        except (AttributeError, ValueError):
            # If parsing fails, count as maximum error
            errors.append(10.0)  # CVSS max is 10
            if detailed_output is not None:
                detailed_results.append({
                    "index": idx,
                    "prompt": prompt,
                    "llm_output": response,
                    "pred": None,
                    "gold": ground_truth,
                    "error": 10.0,
                    "parse_error": True
                })
    
    mad = np.mean(errors) if errors else 0.0  # Mean Absolute Deviation (same as MAE)
    
    print(f"\nMAD (Mean Absolute Deviation): {mad:.4f}")
    
    # Save detailed results
    if detailed_output is not None:
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"Detailed results saved to: {detailed_output}")
    
    return {"mad": mad, "total": len(errors)}


def evaluate_ate(model, tokenizer, dataset, detailed_output=None, **api_kwargs):
    """Evaluate on CTI-ATE (MITRE ATT&CK Technique Extraction)
    
    Uses micro-averaging: pool all predictions across dataset for P/R/F1.
    This gives equal weight to each technique (standard for extraction tasks).
    """
    print("\n" + "="*50)
    print("Evaluating CTI-ATE (Attack Technique Extraction)")
    print("="*50)
    
    # Micro-averaging: global counters
    tp_total = 0
    fp_total = 0
    fn_total = 0
    exact_matches = 0
    detailed_results = []
    
    for idx, sample in enumerate(tqdm(dataset, desc="CTI-ATE")):
        prompt = sample['Prompt']
        ground_truth = sample.get('GT', '').strip()
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        answer = extract_final_answer(response)
        
        # Extract MITRE technique IDs
        pred_techniques = parse_ids_from_text(answer)
        gold_techniques = parse_ids_from_text(ground_truth)
        
        # Compute per-sample TP, FP, FN
        tp = len(pred_techniques & gold_techniques)
        fp = len(pred_techniques - gold_techniques)
        fn = len(gold_techniques - pred_techniques)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        if pred_techniques == gold_techniques:
            exact_matches += 1
        
        # Save detailed result
        if detailed_output is not None:
            detailed_results.append({
                "index": idx,
                "prompt": prompt,
                "llm_output": response,
                "pred": sorted(list(pred_techniques)),
                "gold": sorted(list(gold_techniques)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "exact_match": pred_techniques == gold_techniques
            })
    
    # Compute micro-averaged metrics
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match_rate = exact_matches / len(dataset) if len(dataset) > 0 else 0.0
    
    print(f"\nMicro-averaged metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Exact Match: {exact_match_rate:.4f}")
    print(f"Total TP/FP/FN: {tp_total}/{fp_total}/{fn_total}")
    
    # Save detailed results
    if detailed_output is not None:
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"Detailed results saved to: {detailed_output}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match_rate,
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total
    }


def evaluate_cybermetric(model, tokenizer, dataset_path: str = None, detailed_output=None, **api_kwargs):
    """Evaluate on CyberMetric-500 benchmark
    
    Args:
        dataset_path: Path to CyberMetric-500-v1.json file or URL
    """
    print("\n" + "="*50)
    print("Evaluating CyberMetric-500 (Cybersecurity Knowledge)")
    print("="*50)
    
    # Load CyberMetric dataset
    if dataset_path is None:
        dataset_path = "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-500-v1.json"
    
    if dataset_path.startswith('http'):
        response = requests.get(dataset_path)
        data = response.json()
    else:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    
    questions = data['questions']
    correct = 0
    total = 0
    detailed_results = []
    
    for idx, item in enumerate(tqdm(questions, desc="CyberMetric-500")):
        question = item['question']
        answers = item['answers']  # Dict like {"A": "...", "B": "...", ...}
        correct_answer = item['solution']
        
        # Format options (official CyberMetric format)
        options_str = ', '.join([f"{key}) {value}" for key, value in answers.items()])
        
        # Official CyberMetric prompt format
        # System: "You are a security expert who answers questions."
        # User: "Question: {question}\nOptions: {options}\n\nChoose the correct answer (A, B, C, or D) only. Always return in this format: 'ANSWER: X' "
        prompt = f"Question: {question}\nOptions: {options_str}\n\nChoose the correct answer (A, B, C, or D) only. Always return in this format: 'ANSWER: X' "
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        
        # Extract using official CyberMetric pattern: ANSWER:? [A-D]
        match = re.search(r"ANSWER:?\s*([A-D])", response, re.IGNORECASE)
        answer = match.group(1).upper() if match else extract_final_answer(response, task_type="mcq")
        
        is_correct = answer.upper() == correct_answer.upper()
        if is_correct:
            correct += 1
        
        total += 1
        
        # Save detailed result
        if detailed_output is not None:
            detailed_results.append({
                "index": idx,
                "question": question,
                "answers": answers,
                "llm_output": response,
                "pred": answer,
                "gold": correct_answer,
                "correct": is_correct
            })
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save detailed results
    if detailed_output is not None:
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"Detailed results saved to: {detailed_output}")
    
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_cissp(model, tokenizer, dataset_path: str = None, detailed_output=None, **api_kwargs):
    """Evaluate on CISSP benchmark (cybersecurity certification questions)
    
    Args:
        dataset_path: Path to CISSP JSON file (list of questions with A-D choices)
    """
    print("\n" + "="*50)
    print("Evaluating CISSP (Cybersecurity Certification)")
    print("="*50)
    
    # Load CISSP dataset
    if dataset_path is None:
        raise ValueError("CISSP dataset path must be provided via --cissp_path argument")
    
    if dataset_path.startswith('http'):
        response = requests.get(dataset_path)
        data = response.json()
    else:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    
    # Handle different data structures (list or dict with 'questions'/'items' key)
    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict):
        questions = data.get('questions') or data.get('items') or data.get('data') or []
    else:
        questions = []
    
    correct = 0
    total = 0
    detailed_results = []
    
    for idx, item in enumerate(tqdm(questions, desc="CISSP")):
        # Extract question and choices (flexible key names)
        question = item.get('question') or item.get('Prompt') or ""
        
        # Handle different choice formats
        choices = {}
        if isinstance(item.get('answers'), dict):
            choices = item['answers']
        elif isinstance(item.get('options'), dict):
            choices = item['options']
        elif isinstance(item.get('choices'), list):
            # Convert list to dict with A-D labels
            labels = ['A', 'B', 'C', 'D']
            for label, choice in zip(labels, item['choices']):
                choices[label] = choice
        else:
            # Try direct A/B/C/D keys
            for label in ['A', 'B', 'C', 'D']:
                if label in item:
                    choices[label] = item[label]
        
        # Get correct answer
        correct_answer = ""
        for key in ['solution', 'answer', 'GT', 'correct_answer']:
            if key in item:
                ans = str(item[key]).strip().upper()
                # Extract just the letter
                match = re.search(r'[A-D]', ans)
                if match:
                    correct_answer = match.group(0)
                    break
        
        if not question or not choices or not correct_answer:
            continue
        
        # Format prompt - single answer MCQ format
        options_str = '\n'.join([f"{key}: {value}" for key, value in sorted(choices.items())])
        prompt = f"Question:\n{question}\n\n{options_str}\n\nChoose the single best answer and reply with ONLY one letter: A, B, C, or D."
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        
        # Extract answer (look for A-D in response)
        answer = extract_final_answer(response, task_type="mcq")
        
        is_correct = answer.upper() == correct_answer.upper()
        if is_correct:
            correct += 1
        
        total += 1
        
        # Save detailed result
        if detailed_output is not None:
            detailed_results.append({
                "index": idx,
                "question": question,
                "choices": choices,
                "llm_output": response,
                "pred": answer,
                "gold": correct_answer,
                "correct": is_correct
            })
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save detailed results
    if detailed_output is not None:
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"Detailed results saved to: {detailed_output}")
    
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_seceval(model, tokenizer, dataset_path: str = None, detailed_output=None, **api_kwargs):
    """Evaluate on SecEval benchmark (2126 cybersecurity knowledge questions)
    
    Note: GitHub docs mention 2000 questions, but the actual dataset contains 2126 questions
    
    Args:
        dataset_path: Path to questions.json file or URL
    """
    print("\n" + "="*50)
    print("Evaluating SecEval (Cybersecurity Knowledge)")
    print("="*50)
    
    # Load SecEval dataset
    if dataset_path is None:
        dataset_path = "https://huggingface.co/datasets/XuanwuAI/SecEval/resolve/main/questions.json"
    
    if dataset_path.startswith('http'):
        response = requests.get(dataset_path)
        data = response.json()
    else:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    
    # Official SecEval instruction and few-shot examples
    instruction = "Below are multiple-choice questions concerning cybersecurity. Please select the correct answers and respond with the letters ABCD only."
    
    # Official SecEval few-shot examples
    few_shot_examples = """Question: You are reviewing a piece of code to identify potential vulnerabilities related to NULL pointer dereferences. Which of the following strategies should you consider implementing to mitigate the risk? Choose all that apply. A: Use a programming language that inherently manages pointers and prevents direct memory access. B: Apply input validation to all variables and data stores that receive data from external sources to ensure they are initialized to expected values. C: Explicitly initialize all variables and other data stores, either during declaration or just before the first usage. D: Implement a custom memory management system that automatically assigns a unique identifier to each pointer.
Answer: ABC

Question: Analyze the code snippet provided. Which of the following practices should be implemented in a multithreaded application to prevent NULL pointer dereference vulnerabilities? Select all that apply. A: Ensure that proper locking APIs are used to lock the pointer before the if statement and unlock when it has finished to prevent race conditions. B: Check the return value of all functions that return a value, including network-related functions, and verify that it is non-null before using it. C: Use automated static analysis tools that target this type of weakness, understanding that while not perfect, they can still be effective. D: Verify that a non-nil 'response' is present before deferring 'response.Body.Close()' to handle cases where the 'Do' method returns an error.
Answer: ABCD

"""
    
    correct = 0
    total = 0
    correct_by_topic = {}
    total_by_topic = {}
    detailed_results = []
    
    for idx, item in enumerate(tqdm(data, desc="SecEval")):
        question = item['question']
        choices = item['choices']  # List of choices
        correct_answer = item['answer']  # String like "ABCD" or "AB" (multi-select)
        topics = item.get('topics', ['Unknown'])
        
        # Official SecEval question format: "Question: " + question + " ".join(choices)
        question_text = "Question: " + question + " " + " ".join(choices)
        question_text = question_text.replace("\n", " ")  # Remove newlines as in official script
        
        # Build full prompt with instruction + few-shot + question
        prompt = instruction + "\n\n" + few_shot_examples + question_text + "\n"
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, **api_kwargs)
        
        # Official SecEval extraction logic (from eval.py line 147-153)
        # Strip "Answer:" prefix if present
        llm_output = response
        if "Answer:" in llm_output:
            llm_output = llm_output.replace("Answer:", "")
        # Extract sorted unique letters A-D
        llm_answer = "".join(sorted(list(set(re.findall(r"[A-D]", llm_output)))))
        
        # Normalize correct answer
        correct_answer_normalized = "".join(sorted(correct_answer.upper()))
        
        is_correct = (llm_answer.lower() == correct_answer_normalized.lower())
        
        if is_correct:
            correct += 1
        
        # Track by topic
        for topic in topics:
            if topic not in correct_by_topic:
                correct_by_topic[topic] = 0
                total_by_topic[topic] = 0
            
            if is_correct:
                correct_by_topic[topic] += 1
            total_by_topic[topic] += 1
        
        total += 1
        
        # Save detailed result
        if detailed_output is not None:
            detailed_results.append({
                "index": idx,
                "question": question,
                "choices": choices,
                "llm_output": response,
                "pred": llm_answer,
                "gold": correct_answer_normalized,
                "correct": is_correct,
                "topics": topics
            })
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Print per-topic accuracy
    print("\nPer-Topic Accuracy:")
    topic_results = {}
    for topic in sorted(total_by_topic.keys()):
        topic_acc = correct_by_topic[topic] / total_by_topic[topic]
        topic_results[topic] = {
            "accuracy": topic_acc,
            "correct": correct_by_topic[topic],
            "total": total_by_topic[topic]
        }
        print(f"  {topic}: {topic_acc:.4f} ({correct_by_topic[topic]}/{total_by_topic[topic]})")
    
    # Save detailed results
    if detailed_output is not None:
        with open(detailed_output, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"Detailed results saved to: {detailed_output}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_topic": topic_results
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate on CTI-Bench and Additional Benchmarks")
    
    # Model loading options
    parser.add_argument("--model_path", type=str, help="Path to model (base or fine-tuned) - for local inference")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name (required for LoRA)")
    parser.add_argument("--is_base", action="store_true", help="Evaluate base model (pre-training)")
    
    # API endpoint options
    parser.add_argument("--use_api", action="store_true", help="Use API endpoint instead of local model")
    parser.add_argument("--api_endpoint", type=str, help="OpenAI-compatible API endpoint (e.g., http://IP:7799/v1/chat/completions)")
    parser.add_argument("--api_model", type=str, help="Model name for API endpoint")
    parser.add_argument("--api_key", type=str, default="", help="API key if needed (leave empty for local vLLM)")
    
    # Evaluation options
    parser.add_argument("--tasks", nargs="+", default=["mcq", "rcm", "vsp", "ate"], 
                       help="Tasks to evaluate (mcq, rcm, vsp, ate, cybermetric, seceval, cissp)")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples per task (for testing)")
    
    # Dataset paths for external benchmarks
    parser.add_argument("--cissp_path", type=str, default=None, help="Path to CISSP dataset JSON file")
    # Collected-output evaluation mode
    parser.add_argument("--input_jsonl", type=str, default=None,
                        help="Collected model-output JSONL from inference script")
    parser.add_argument("--evaluate_collected", action="store_true",
                        help="Evaluate already-collected outputs instead of running inference")
    parser.add_argument("--detailed_output", type=str, default=None,
                        help="Optional detailed JSONL output path for collected evaluation")
    parser.add_argument("--alias_dict_path", type=str, default=None,
                        help="Path to CTI-Bench alias_dict.pickle for TAA")
    parser.add_argument("--related_dict_path", type=str, default=None,
                        help="Path to CTI-Bench related_dict.pickle for TAA")
    parser.add_argument("--ctibench_cvss_prefix", type=str, default="CVSS:3.0/",
                        help="CTI-VSP prefix. Original notebook uses CVSS:3.0/")
    parser.add_argument("--benchmark", type=str, default="ctibench",
                        choices=[
                            "ctibench", "athenabench", "cybermetric", "seceval",
                            "redsage", "secure", "cissp", "secbench", "mmlu_cs",
                        ],
                        help="Which collected-output benchmark evaluator to use")
    parser.add_argument("--athena_alias_csv_path", type=str, default=None,
                        help="Path to AthenaBench aliases CSV for TAA")
    parser.add_argument("--athena_related_csv_path", type=str, default=None,
                        help="Path to AthenaBench related groups CSV for TAA")
    parser.add_argument("--athena_vsp_denominator", type=float, default=7.7,
                        help="AthenaBench VSP MAD denominator; official default is 7.7")
    parser.add_argument("--ctibench_taa_gt_tsv", type=str, default=None,
                        help="Path to CTI-TAA ground-truth TSV (evaluation/responses/cti-taa-responses.tsv). "
                             "Required because the public dataset has no GT column.")
    

    
    args = parser.parse_args()
    if args.evaluate_collected:
        if not args.input_jsonl:
            parser.error("--input_jsonl is required with --evaluate_collected")

        if args.benchmark == "ctibench":
            result = evaluate_collected_ctibench(
                input_jsonl=args.input_jsonl,
                detailed_output=args.detailed_output,
                alias_dict_path=args.alias_dict_path,
                related_dict_path=args.related_dict_path,
                cvss_prefix=args.ctibench_cvss_prefix,
                taa_gt_tsv_path=args.ctibench_taa_gt_tsv,
            )

        elif args.benchmark == "athenabench":
            result = evaluate_collected_athenabench(
                input_jsonl=args.input_jsonl,
                detailed_output=args.detailed_output,
                alias_csv_path=args.athena_alias_csv_path,
                related_csv_path=args.athena_related_csv_path,
                vsp_denominator=args.athena_vsp_denominator,
            )

        elif args.benchmark == "cybermetric":
            result = evaluate_collected_cybermetric(
                input_jsonl=args.input_jsonl,
                detailed_output=args.detailed_output,
            )
        elif args.benchmark == "seceval":
            result = evaluate_collected_seceval(
                input_jsonl=args.input_jsonl,
                detailed_output=args.detailed_output,
            )
        elif args.benchmark == "redsage":
            result = evaluate_collected_redsage(
                input_jsonl=args.input_jsonl,
                detailed_output=args.detailed_output,
            )

        elif args.benchmark == "secure":
            result = evaluate_collected_secure(
                input_jsonl=args.input_jsonl,
                detailed_output=args.detailed_output,
            )

        elif args.benchmark == "cissp":
            result = evaluate_collected_cissp(
                input_jsonl=args.input_jsonl,
                detailed_output=args.detailed_output,
            )

        elif args.benchmark == "secbench":
            result = evaluate_collected_secbench_mcq(
                input_jsonl=args.input_jsonl,
                detailed_output=args.detailed_output,
            )

        elif args.benchmark == "mmlu_cs":
            result = evaluate_collected_mmlu_cs(
                input_jsonl=args.input_jsonl,
                detailed_output=args.detailed_output,
            )

        else:
            parser.error(f"Unsupported benchmark: {args.benchmark}")


        result = build_unified_result(
            benchmark=args.benchmark,
            original_result=result,
            input_jsonl=args.input_jsonl,
            detailed_output=args.detailed_output,
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        return
    
    
    # Validate arguments
    if args.use_api:
        if not args.api_endpoint or not args.api_model:
            parser.error("--api_endpoint and --api_model are required when --use_api is set")
        model = None
        tokenizer = None
    else:
        if not args.model_path:
            parser.error("--model_path is required for local inference (or use --use_api)")
        # Load model for local inference
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model, args.is_base)
    
    # Generate output filename from model path if not specified
    if args.output is None:
        if args.use_api:
            model_name = args.api_model.rstrip('/').split('/')[-1]
        else:
            model_name = args.model_path.rstrip('/').split('/')[-1]
        # Sanitize filename
        model_name = model_name.replace('/', '-').replace('\\', '-')
        args.output = f"eval_results_{model_name}.json"
    
    # Create detailed output directory
    detailed_dir = args.output.replace('.json', '_detailed')
    os.makedirs(detailed_dir, exist_ok=True)
    print(f"Detailed results will be saved to: {detailed_dir}/")
    
    # Prepare API kwargs for generate_response
    api_kwargs = {
        'use_api': args.use_api,
        'api_endpoint': args.api_endpoint,
        'api_model': args.api_model,
        'api_key': args.api_key
    }
    
    # Run evaluations
    results = {
        "model_path": args.model_path if not args.use_api else args.api_model,
        "evaluation_mode": "api" if args.use_api else "local",
        "is_base_model": args.is_base,
        "tasks": {}
    }
    
    task_map = {
        "mcq": ("AI4Sec/cti-bench", "cti-mcq", evaluate_mcq),
        "rcm": ("AI4Sec/cti-bench", "cti-rcm", evaluate_rcm),
        "vsp": ("AI4Sec/cti-bench", "cti-vsp", evaluate_vsp),
        "ate": ("AI4Sec/cti-bench", "cti-ate", evaluate_ate),
        "cybermetric": (None, None, evaluate_cybermetric),
        "seceval": (None, None, evaluate_seceval),
        "cissp": (None, None, evaluate_cissp),
    }
    
    for task_name in args.tasks:
        if task_name not in task_map:
            print(f"Unknown task: {task_name}, skipping...")
            continue
        
        dataset_name, subset_name, eval_fn = task_map[task_name]
        detailed_output = os.path.join(detailed_dir, f"{task_name}_detailed.jsonl")
        
        # Special handling for external JSON benchmarks (CyberMetric, SecEval, CISSP)
        if task_name in ["cybermetric", "seceval", "cissp"]:
            # Determine dataset path
            if task_name == "cissp":
                dataset_path = args.cissp_path
            else:
                dataset_path = None  # Uses default URLs
            
            task_results = eval_fn(model, tokenizer, dataset_path=dataset_path, 
                                 detailed_output=detailed_output, **api_kwargs)
            results["tasks"][task_name.upper()] = task_results
        else:
            print(f"\nLoading {subset_name} dataset...")
            dataset = load_dataset(dataset_name, subset_name, split="test")
            
            if args.max_samples:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            
            # Run evaluation with detailed output
            task_results = eval_fn(model, tokenizer, dataset, detailed_output=detailed_output, **api_kwargs)
            results["tasks"][task_name.upper()] = task_results
    
    # Save results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(json.dumps(results, indent=2))
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    print(f"Detailed results directory: {detailed_dir}/")


if __name__ == "__main__":
    main()
