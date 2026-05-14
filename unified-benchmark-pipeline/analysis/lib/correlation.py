"""Pairwise correlation utilities with bootstrap and permutation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def kendall_tau(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Kendall τ-b with associated two-sided p-value (asymptotic).

    Uses scipy on first call (for the p-value) but for repeated bootstrap calls
    we have a faster pure-numpy version below.
    """
    res = stats.kendalltau(x, y, variant="b", nan_policy="omit")
    return float(res.statistic), float(res.pvalue)


def kendall_tau_b_fast(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Kendall τ-b without p-value, vectorised in NumPy.

    For the small n we use here (10 models per task) this is ~10× faster than
    scipy.stats.kendalltau and contributes most of the speed-up in bootstrap.
    """
    n = len(x)
    if n < 2:
        return float("nan")
    xi, xj = np.meshgrid(x, x, indexing="ij")
    yi, yj = np.meshgrid(y, y, indexing="ij")
    dx = np.sign(xi - xj)
    dy = np.sign(yi - yj)
    triu = np.triu(np.ones_like(dx, dtype=bool), k=1)
    same = (dx * dy)[triu]
    concordant = (same > 0).sum()
    discordant = (same < 0).sum()
    tied_x = ((dx == 0) & (dy != 0))[triu].sum()
    tied_y = ((dx != 0) & (dy == 0))[triu].sum()
    n_pairs = n * (n - 1) // 2
    n1 = n_pairs - tied_x  # pairs not tied on x
    n2 = n_pairs - tied_y
    denom = float(np.sqrt(max(1.0, n1) * max(1.0, n2)))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    res = stats.spearmanr(x, y, nan_policy="omit")
    return float(res.statistic), float(res.pvalue)


def pearson_r(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    res = stats.pearsonr(x, y)
    return float(res.statistic), float(res.pvalue)


def all_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[float, float]]:
    return {
        "kendall": kendall_tau(x, y),
        "spearman": spearman_rho(x, y),
        "pearson": pearson_r(x, y),
    }


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = "kendall",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Resample model-pairs with replacement; return (point, lower, upper).

    Uses the fast Kendall implementation when metric=='kendall'.
    """
    if metric == "kendall":
        point = kendall_tau_b_fast(x, y)
        n = len(x)
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            if len(set(idx)) < 3:
                continue
            v = kendall_tau_b_fast(x[idx], y[idx])
            if not np.isnan(v):
                samples.append(v)
        if not samples:
            return point, float("nan"), float("nan")
        lo = float(np.percentile(samples, 100 * (1 - confidence) / 2))
        hi = float(np.percentile(samples, 100 * (1 + confidence) / 2))
        return point, lo, hi

    fn = {"spearman": spearman_rho, "pearson": pearson_r}[metric]
    point, _ = fn(x, y)
    n = len(x)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if len(set(idx)) < 3:
            continue
        try:
            v, _ = fn(x[idx], y[idx])
            if not np.isnan(v):
                samples.append(v)
        except Exception:
            continue
    if not samples:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(samples, 100 * (1 - confidence) / 2))
    hi = float(np.percentile(samples, 100 * (1 + confidence) / 2))
    return point, lo, hi


def permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = "kendall",
    n_resamples: int = 1000,
    seed: int = 0,
) -> float:
    """Two-sided p-value: P(|τ_perm| >= |τ_obs|) under random shuffles of y."""
    fn = {"kendall": kendall_tau, "spearman": spearman_rho, "pearson": pearson_r}[metric]
    obs, _ = fn(x, y)
    if np.isnan(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    y2 = y.copy()
    geq = 0
    valid = 0
    for _ in range(n_resamples):
        rng.shuffle(y2)
        v, _ = fn(x, y2)
        if not np.isnan(v):
            valid += 1
            if abs(v) >= abs(obs) - 1e-12:
                geq += 1
    if valid == 0:
        return float("nan")
    return (geq + 1) / (valid + 1)


def pairwise_matrix(
    M: np.ndarray, fn=kendall_tau
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (correlation_matrix, pvalue_matrix), both shape (n_rows, n_rows)."""
    n = M.shape[0]
    C = np.full((n, n), np.nan)
    P = np.full((n, n), np.nan)
    for i in range(n):
        C[i, i] = 1.0
        P[i, i] = 0.0
        for j in range(i + 1, n):
            v, p = fn(M[i], M[j])
            C[i, j] = C[j, i] = v
            P[i, j] = P[j, i] = p
    return C, P


def partial_kendall(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> float:
    """Partial Kendall τ via Pearson-on-ranks of residuals.

    Approximates τ_{xy.z} using the standard partial-correlation formula on ranks.
    """
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt(max(1e-12, (1 - rxz**2) * (1 - ryz**2)))
    return float((rxy - rxz * ryz) / denom)
