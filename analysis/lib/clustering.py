"""Hierarchical clustering and PCA helpers."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def signed_distance(C: np.ndarray) -> np.ndarray:
    """1 - τ on a symmetric, diagonal-1 correlation matrix."""
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)
    D = 0.5 * (D + D.T)
    return D


def linkage_matrix(D: np.ndarray, method: str = "average") -> np.ndarray:
    return hierarchy.linkage(squareform(D, checks=False), method=method)


def leaves_order(Z: np.ndarray) -> List[int]:
    return list(hierarchy.leaves_list(Z))


def fcluster_at_k(Z: np.ndarray, k: int) -> np.ndarray:
    return hierarchy.fcluster(Z, t=k, criterion="maxclust")


def pca(
    M: np.ndarray, standardise: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (components, explained_variance_ratio, projected).

    `M` is shape (n_items, n_features). For our use, items = sub-tasks,
    features = model accuracies.
    """
    X = M.astype(float).copy()
    if standardise:
        mu = np.nanmean(X, axis=0, keepdims=True)
        sd = np.nanstd(X, axis=0, ddof=0, keepdims=True)
        sd[sd < 1e-12] = 1.0
        X = (X - mu) / sd
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = (S ** 2) / max(1, X.shape[0] - 1)
    ratio = var / var.sum()
    proj = U * S
    return Vt, ratio, proj


def parallel_analysis_threshold(
    M: np.ndarray,
    n_resamples: int = 200,
    standardise: bool = True,
    seed: int = 0,
    percentile: float = 95.0,
) -> np.ndarray:
    """Return the percentile-th explained-variance ratios of null random matrices."""
    rng = np.random.default_rng(seed)
    n_rows, n_cols = M.shape
    eigs = []
    for _ in range(n_resamples):
        R = rng.standard_normal((n_rows, n_cols))
        if standardise:
            R = (R - R.mean(axis=0)) / R.std(axis=0, ddof=0)
        R = R - R.mean(axis=0)
        s = np.linalg.svd(R, compute_uv=False)
        ev = (s ** 2) / max(1, n_rows - 1)
        eigs.append(ev / ev.sum())
    eigs = np.array(eigs)
    return np.percentile(eigs, percentile, axis=0)


def effective_dimensions(
    explained_ratio: np.ndarray,
    null_thresholds: np.ndarray = None,
    target: float = 0.90,
) -> dict:
    """Return effective dimensionality summaries: cumulative-variance and Horn.

    `null_thresholds` should be on the same scale as `explained_ratio`
    (i.e. percentile of the null distribution of variance-ratios).
    """
    cum = np.cumsum(explained_ratio)
    n_for_target = int(np.searchsorted(cum, target) + 1)
    horn = None
    if null_thresholds is not None:
        horn = int(np.sum(explained_ratio > null_thresholds))
    return {
        "cumulative_variance": cum.tolist(),
        "n_for_target": n_for_target,
        "target": target,
        "horn_n": horn,
    }
