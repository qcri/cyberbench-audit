"""Vote engines for majority / weighted / topk / acceptance experiments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VoteResult:
    gold: str
    majority_prediction: Optional[str]
    agreement_count: float          # weighted-or-unweighted vote total for the majority
    agreement_total: float          # sum of weights of all voting models
    agreement_fraction: float       # agreement_count / agreement_total
    n_models_with_data: int
    agreeing_models: List[str] = field(default_factory=list)
    all_predictions: Dict[str, str] = field(default_factory=dict)


def vote(
    predictions: Dict[str, str],
    gold: str,
    weights: Optional[Dict[str, float]] = None,
) -> Optional[VoteResult]:
    """Compute weighted majority vote.

    `predictions`: model_name -> normalized prediction string (already filtered to
    non-null preds).
    `weights`: optional model_name -> positive float; defaults to 1.0 each.
    """
    if not predictions:
        return None

    weights = weights or {m: 1.0 for m in predictions}

    pred_weight = defaultdict(float)  # type: Dict[str, float]
    pred_models = defaultdict(list)   # type: Dict[str, List[str]]
    total_weight = 0.0
    for model, pred in predictions.items():
        w = weights.get(model, 0.0)
        if w <= 0:
            continue
        pred_weight[pred] += w
        pred_models[pred].append(model)
        total_weight += w

    if total_weight == 0 or not pred_weight:
        return None

    majority_pred, majority_w = max(pred_weight.items(), key=lambda kv: kv[1])
    return VoteResult(
        gold=gold,
        majority_prediction=majority_pred,
        agreement_count=majority_w,
        agreement_total=total_weight,
        agreement_fraction=majority_w / total_weight,
        n_models_with_data=len(predictions),
        agreeing_models=sorted(pred_models[majority_pred]),
        all_predictions=dict(predictions),
    )


def is_flagged(result: VoteResult, threshold: float) -> bool:
    return (
        result.majority_prediction is not None
        and result.majority_prediction != result.gold
        and result.agreement_fraction >= threshold
    )


def linear_rank_weights(model_accuracies: Dict[str, float]) -> Dict[str, float]:
    """Linear weighting: w_i = (N - rank + 1) / N. Rank 1 = highest accuracy."""
    ordered = sorted(model_accuracies.items(), key=lambda kv: -kv[1])
    n = len(ordered)
    if n == 0:
        return {}
    return {m: (n - i) / n for i, (m, _) in enumerate(ordered)}


def harmonic_rank_weights(model_accuracies: Dict[str, float]) -> Dict[str, float]:
    """Harmonic weighting: w_i = 1 / rank. Rank 1 = highest accuracy."""
    ordered = sorted(model_accuracies.items(), key=lambda kv: -kv[1])
    return {m: 1.0 / (i + 1) for i, (m, _) in enumerate(ordered)}


def topk_filter(
    predictions: Dict[str, str],
    model_accuracies: Dict[str, float],
    k: int,
) -> Dict[str, str]:
    """Keep predictions only from top-k models by accuracy."""
    if k >= len(model_accuracies):
        return dict(predictions)
    top = {m for m, _ in sorted(model_accuracies.items(), key=lambda kv: -kv[1])[:k]}
    return {m: p for m, p in predictions.items() if m in top}


def acceptance_filter(
    predictions: Dict[str, str],
    model_accuracies: Dict[str, float],
    cutoff: float,
) -> Dict[str, str]:
    """Keep predictions only from models whose accuracy >= cutoff."""
    eligible = {m for m, a in model_accuracies.items() if a >= cutoff}
    return {m: p for m, p in predictions.items() if m in eligible}
