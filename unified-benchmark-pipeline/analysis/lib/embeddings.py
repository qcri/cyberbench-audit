"""Question-text loader, boilerplate stripper, and sentence-transformers wrapper."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from analysis.lib.loaders import iter_jsonl, responses_path


# Per-task scaffolding to strip. Patterns are matched and removed (case
# sensitive — they match exactly what the inference scripts injected).
_BOILERPLATE_PATTERNS = [
    # MCQ A-D variants
    r"Choose the correct answer \(A, B, C, or D\) only\.\s*Always return in this format:\s*'ANSWER:\s*X'\.?",
    r"Return your answer as a single uppercase letter:\s*A,\s*B,\s*C,\s*or\s*D\.?\s*\*?\*?Important:?\*?\*?\s*The last line of your answer should contain only the single letter corresponding to the best option, with no additional text\.?",
    r"Answer with the option letter\s*\(A,\s*B,\s*C,\s*D\)\s*from the given choices directly\.?",
    r"Answer with the option letter from the given choices directly\.?",
    # ID-extraction
    r"Ensure the final line contains only the IDs for the main techniques,\s*separated by commas,\s*excluding any subtechnique IDs\.?",
    r"Provide.*MITRE.*technique IDs.*comma.*separated.*",
    # CVSS
    r"Ensure the final line of your response contains only the CVSS v3 Vector String in the following format:[^\n]*Example format:\s*CVSS:3\.1/[^\s]+",
    r"Ensure the final line of your response contains only the CVSS v3 Vector String[^\n]*",
    # TFX
    r"Return your answer as either T \(for True\) or F \(for False\)\.\s*If you do not know the answer,\s*return X\.\s*Provide only the letter corresponding to your choice\s*\(T,\s*F,\s*or X\)\s*without any additional text or explanations\.?",
    # CWE
    r"Ensure the last line of your response contains only the CWE ID\.?",
    # Generic trailing chat-format leftovers
    r"\nAnswer:\s*$",
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _BOILERPLATE_PATTERNS]


def strip_boilerplate(text: str) -> str:
    """Remove common task-instruction scaffolding from a question prompt."""
    if not text:
        return ""
    out = text
    for pat in _COMPILED:
        out = pat.sub("", out)
    return out.strip()


def prompt_to_text(prompt) -> str:
    """Coerce a `prompt` field (str or list-of-messages) to one string."""
    if isinstance(prompt, list):
        # chat format: keep user content only
        parts = []
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content", "")
                if c:
                    parts.append(c)
        return "\n\n".join(parts)
    return str(prompt or "")


def iter_task_questions(
    outputs_root: Path,
    task: str,
    preferred_models: List[str] = ("GPT-5.4", "Fanar-2-27B-Instruct"),
) -> Iterator[Tuple[str, str]]:
    """Yield (sample_index, cleaned_text) pairs for a task.

    Reads from the first preferred model whose responses file exists.
    """
    for m in preferred_models:
        path = responses_path(outputs_root, m, task)
        if path.exists():
            for sample in iter_jsonl(path):
                idx = str(sample.get("index", ""))
                if not idx:
                    continue
                raw = prompt_to_text(sample.get("prompt", ""))
                cleaned = strip_boilerplate(raw)
                if cleaned:
                    yield idx, cleaned
            return
    # nothing found
    return


def encode_batch(
    texts: List[str],
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 64,
    device: Optional[str] = None,
    normalize: bool = True,
):
    """Encode a list of texts; returns (n, dim) numpy float32 array.

    Lazy-imports sentence-transformers / torch so the lib loads even
    without those deps.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return vectors
