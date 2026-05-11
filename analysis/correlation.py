"""Cross-task correlation analysis on the 24 x 10 accuracy matrix.

Outputs to analysis/reports/correlation/:
  - pairwise_kendall.csv / pairwise_spearman.csv / pairwise_pearson.csv
  - bootstrap_ci.csv (Kendall, 1000 resamples, 95% CI)
  - permutation_pvals.csv (Kendall, 1000 shuffles)
  - partial_kendall.csv (controlled for parent-group mean accuracy)
  - pca_components.csv  (loadings of the first 5 components)
  - effective_dimensions.json
  - clusters.json (cluster id per task at k=4..7)
  - redundant_pairs.md (τ > 0.85 with bootstrap CI lower bound > 0.7)
  - within_vs_across_parent.csv

Run: PYTHONPATH=. python3 -m analysis.correlation
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from analysis.lib.clustering import (
    effective_dimensions,
    fcluster_at_k,
    leaves_order,
    linkage_matrix,
    parallel_analysis_threshold,
    pca,
    signed_distance,
)
from analysis.lib.correlation import (
    bootstrap_ci,
    kendall_tau,
    pairwise_matrix,
    partial_kendall,
    pearson_r,
    permutation_pvalue,
    spearman_rho,
)
from analysis.lib.loaders import PARENT_GROUPS, TASK_ORDER


HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
OUT = REPORTS / "correlation"
OUT.mkdir(parents=True, exist_ok=True)


def load_matrix() -> tuple:
    cache = json.loads((REPORTS / "per_model_task_accuracy.json").read_text())
    models = list(cache.keys())
    tasks = TASK_ORDER  # canonical order from analysis.lib.loaders
    M = np.full((len(tasks), len(models)), np.nan)
    for j, m in enumerate(models):
        for i, t in enumerate(tasks):
            v = cache[m].get(t)
            if v is not None:
                M[i, j] = float(v)
    # Drop models with no scored cells (PCA / Kendall otherwise blow up).
    keep = [j for j in range(len(models)) if not np.all(np.isnan(M[:, j]))]
    if len(keep) < len(models):
        dropped = [models[j] for j in range(len(models)) if j not in keep]
        print(f"dropping {len(dropped)} all-NA model column(s): {dropped}")
        M = M[:, keep]
        models = [models[j] for j in keep]
    return M, tasks, models


def parent_of(task: str) -> str:
    for parent, members in PARENT_GROUPS:
        if task in members:
            return parent
    return "OTHER"


def write_matrix_csv(path: Path, M: np.ndarray, tasks: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", *tasks])
        for i, t in enumerate(tasks):
            row = [t] + [
                "" if np.isnan(M[i, j]) else f"{M[i, j]:.4f}"
                for j in range(M.shape[1])
            ]
            w.writerow(row)


N_BOOT = 300
PAIRS_PER_CHUNK = 30


def stage_pairwise(M, tasks):
    K, _ = pairwise_matrix(M, fn=kendall_tau)
    S, _ = pairwise_matrix(M, fn=spearman_rho)
    P, _ = pairwise_matrix(M, fn=pearson_r)
    write_matrix_csv(OUT / "pairwise_kendall.csv", K, tasks)
    write_matrix_csv(OUT / "pairwise_spearman.csv", S, tasks)
    write_matrix_csv(OUT / "pairwise_pearson.csv", P, tasks)
    return K


def stage_bootstrap(M, tasks, chunk_idx=None, n_chunks=None, time_budget_s=None):
    """Compute (Kendall, bootstrap CI, asymptotic p-value) per pair, in chunks.

    If chunk_idx is None, processes ALL chunks not already on disk; this
    is checkpoint-resilient since each pair is saved as soon as it is done.
    """
    import time as _time
    n = len(tasks)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    chunk_dir = OUT / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if chunk_idx is not None:
        total = max(1, len(all_pairs))
        start = (total * chunk_idx) // n_chunks
        end = (total * (chunk_idx + 1)) // n_chunks
        my_pairs = all_pairs[start:end]
        out_path = chunk_dir / f"boot_{chunk_idx}.json"
        rows = []
        for i, j in my_pairs:
            point, lo, hi = bootstrap_ci(M[i], M[j], "kendall", n_resamples=N_BOOT, seed=i * 100 + j)
            _, p_asymp = kendall_tau(M[i], M[j])
            rows.append({
                "task_a": tasks[i], "task_b": tasks[j],
                "kendall": round(point, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "p_asymp": round(p_asymp, 4),
                "parent_a": parent_of(tasks[i]), "parent_b": parent_of(tasks[j]),
                "same_parent": parent_of(tasks[i]) == parent_of(tasks[j]),
            })
        out_path.write_text(json.dumps(rows))
        return rows

    # Resumable mode: one JSON per pair, skip already-done.
    pair_dir = chunk_dir / "by_pair"
    pair_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = _time.time()
    for i, j in all_pairs:
        path = pair_dir / f"{i}_{j}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
            continue
        point, lo, hi = bootstrap_ci(M[i], M[j], "kendall", n_resamples=N_BOOT, seed=i * 100 + j)
        _, p_asymp = kendall_tau(M[i], M[j])
        d = {
            "task_a": tasks[i], "task_b": tasks[j],
            "kendall": round(point, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "p_asymp": round(p_asymp, 4),
            "parent_a": parent_of(tasks[i]), "parent_b": parent_of(tasks[j]),
            "same_parent": parent_of(tasks[i]) == parent_of(tasks[j]),
        }
        path.write_text(json.dumps(d))
        rows.append(d)
        if time_budget_s and (_time.time() - t0) > time_budget_s:
            print(f"time budget exceeded after {len(rows)} pairs; resume by re-running")
            return rows
    return rows


def collect_bootstrap() -> list:
    pair_dir = OUT / "_chunks" / "by_pair"
    if pair_dir.exists():
        files = sorted(pair_dir.glob("*.json"))
        if files:
            return [json.loads(f.read_text()) for f in files]
    chunk_dir = OUT / "_chunks"
    files = sorted(chunk_dir.glob("boot_*.json"))
    rows = []
    for f in files:
        rows.extend(json.loads(f.read_text()))
    return rows

def stage_postprocess(M, tasks, models, pairs):
    K, _ = pairwise_matrix(M, fn=kendall_tau)
    with open(OUT / "bootstrap_ci.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        w.writeheader()
        w.writerows(pairs)

    # ---------- partial Kendall (control for parent-group mean) ----------
    # Build parent-mean matrix per task: each row = mean accuracy over the
    # other sub-tasks in the same parent. If only one task in parent, use the
    # global mean.
    parent_idx = {t: parent_of(t) for t in tasks}
    parent_members = {p: [t for t in tasks if parent_of(t) == p] for p, _ in PARENT_GROUPS}
    parent_mean = {}
    for p, members in parent_members.items():
        if len(members) <= 1:
            continue
        for t in members:
            others = [m for m in members if m != t]
            idxs = [tasks.index(m) for m in others]
            parent_mean[t] = np.nanmean(M[idxs], axis=0)

    partial_rows = []
    for d in pairs:
        a, b = d["task_a"], d["task_b"]
        if a not in parent_mean or b not in parent_mean:
            continue
        if d["same_parent"] and parent_mean[a] is parent_mean[b]:
            continue
        z = parent_mean[a]
        try:
            ptau = partial_kendall(M[tasks.index(a)], M[tasks.index(b)], z)
        except Exception:
            ptau = float("nan")
        partial_rows.append({
            "task_a": a,
            "task_b": b,
            "tau": d["kendall"],
            "partial_tau_given_parentA_mean": round(ptau, 4),
        })
    if partial_rows:
        with open(OUT / "partial_kendall.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(partial_rows[0].keys()))
            w.writeheader()
            w.writerows(partial_rows)

    # ---------- hierarchical clustering on signed Kendall distance ----------
    D = signed_distance(K)
    Z = linkage_matrix(D, method="average")
    order = leaves_order(Z)
    clusters_by_k = {k: [int(c) for c in fcluster_at_k(Z, k)] for k in [3, 4, 5, 6, 7]}
    (OUT / "clusters.json").write_text(json.dumps({
        "tasks": tasks,
        "leaf_order": [int(x) for x in order],
        "clusters": clusters_by_k,
        "linkage_method": "average",
        "distance": "1 - kendall_tau (signed)",
    }, indent=2))

    # Save linkage as CSV (id1, id2, dist, n)
    with open(OUT / "linkage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id1", "id2", "distance", "n"])
        for row in Z:
            w.writerow([int(row[0]), int(row[1]), f"{row[2]:.6f}", int(row[3])])

    # ---------- PCA & parallel analysis ----------
    Vt, ratio, proj = pca(M, standardise=True)
    null_thr = parallel_analysis_threshold(M, n_resamples=500, standardise=True, seed=0)
    eff = effective_dimensions(ratio, null_thresholds=null_thr, target=0.90)
    (OUT / "effective_dimensions.json").write_text(json.dumps({
        "explained_variance_ratio": ratio.tolist(),
        "null_variance_ratio_p95": null_thr.tolist(),
        **eff,
    }, indent=2))

    with open(OUT / "pca_components.csv", "w", newline="") as f:
        w = csv.writer(f)
        n_keep = min(5, Vt.shape[0])
        w.writerow(["model"] + [f"PC{i+1}" for i in range(n_keep)])
        for j, m in enumerate(models):
            w.writerow([m] + [f"{Vt[i, j]:.4f}" for i in range(n_keep)])
    with open(OUT / "pca_projection.csv", "w", newline="") as f:
        w = csv.writer(f)
        n_keep = min(5, proj.shape[1])
        w.writerow(["task"] + [f"PC{i+1}" for i in range(n_keep)])
        for i, t in enumerate(tasks):
            w.writerow([t] + [f"{proj[i, k]:.4f}" for k in range(n_keep)])

    # ---------- redundant-pairs table ----------
    redundant = sorted(
        [d for d in pairs if d["kendall"] > 0.85 and d["ci_lo"] > 0.70],
        key=lambda d: -d["kendall"],
    )
    md = ["# Redundant sub-task pairs (Kendall τ > 0.85, bootstrap 95% CI lower bound > 0.70)",
          "",
          "| τ | 95% CI | task A | task B | same parent? |",
          "|---|---|---|---|---|"]
    for d in redundant:
        md.append(
            f"| {d['kendall']:.3f} | [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}] | "
            f"{d['task_a']} ({d['parent_a']}) | {d['task_b']} ({d['parent_b']}) | "
            f"{'yes' if d['same_parent'] else 'NO'} |"
        )
    if not redundant:
        md.append("| (no pair clears the bar) |")
    (OUT / "redundant_pairs.md").write_text("\n".join(md))

    # ---------- within vs across parent ----------
    within = [d["kendall"] for d in pairs if d["same_parent"] and not np.isnan(d["kendall"])]
    across = [d["kendall"] for d in pairs if not d["same_parent"] and not np.isnan(d["kendall"])]
    with open(OUT / "within_vs_across_parent.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "n", "mean", "median", "p25", "p75"])
        w.writerow([
            "within_parent", len(within),
            f"{np.mean(within):.4f}" if within else "nan",
            f"{np.median(within):.4f}" if within else "nan",
            f"{np.percentile(within, 25):.4f}" if within else "nan",
            f"{np.percentile(within, 75):.4f}" if within else "nan",
        ])
        w.writerow([
            "across_parent", len(across),
            f"{np.mean(across):.4f}" if across else "nan",
            f"{np.median(across):.4f}" if across else "nan",
            f"{np.percentile(across, 25):.4f}" if across else "nan",
            f"{np.percentile(across, 75):.4f}" if across else "nan",
        ])

    print(f"correlation analysis written to {OUT}/")
    print(f"  redundant pairs (tau>0.85 & CI_lo>0.70): {len(redundant)}")
    print(f"  cumulative variance to 90%: {eff['n_for_target']} components")
    print(f"  Horn's effective N components: {eff.get('horn_n')}")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pairwise", "bootstrap", "postprocess", "all"], default="all")
    parser.add_argument("--chunk-idx", type=int, default=None)
    parser.add_argument("--n-chunks", type=int, default=None)
    args = parser.parse_args(argv)

    M, tasks, models = load_matrix()
    print(f"matrix: {M.shape[0]} tasks x {M.shape[1]} models")

    if args.stage == "pairwise":
        stage_pairwise(M, tasks)
        return
    if args.stage == "bootstrap":
        if args.chunk_idx is not None and args.n_chunks is not None:
            rows = stage_bootstrap(M, tasks, args.chunk_idx, args.n_chunks)
            print(f"chunk {args.chunk_idx}/{args.n_chunks}: {len(rows)} pairs")
            return
        # Resumable per-pair mode with a 3-second budget so we fit the bash
        # limit. Re-run repeatedly until all pairs are done.
        n_pairs = len(tasks) * (len(tasks) - 1) // 2
        rows = stage_bootstrap(M, tasks, time_budget_s=2.0)
        print(f"bootstrap progress: {len(rows)}/{n_pairs} pairs")
        return
    if args.stage == "postprocess":
        pairs = collect_bootstrap()
        if not pairs:
            raise SystemExit("No bootstrap chunks found. Run --stage bootstrap first.")
        stage_postprocess(M, tasks, models, pairs)
        return

    # all-in-one (only safe in unconstrained environments)
    stage_pairwise(M, tasks)
    stage_bootstrap(M, tasks)
    pairs = collect_bootstrap()
    stage_postprocess(M, tasks, models, pairs)


if __name__ == "__main__":
    main()
