"""Centroid-similarity matrix + EVoC clustering + comparison vs Kendall τ.

Reads:
  reports/embeddings/embeddings/<task>.npz   (from analysis.embed)
  reports/correlation/pairwise_kendall.csv   (from analysis.correlation)

Writes (under reports/embeddings/):
  centroid_similarity.csv        22 x 22 cosine sim between sub-task centroids
  centroid_vs_kendall.csv        per-pair: semantic_sim, kendall_tau, quadrant
  evoc_labels.csv                per-question (task, idx) -> cluster id
  cluster_task_purity.csv        per-cluster: dominant_task, purity, size, top-3
  task_overlap_matrix.csv        22 x 22 EVoC-derived shared-cluster ratio
  overlap_pairs.md               ranked list: pairs with biggest content overlap
  mantel.json                    Mantel-style correlation between centroid-sim
                                 and Kendall-tau matrices
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analysis.lib.evoc_cluster import cluster as evoc_cluster
from analysis.lib.loaders import TASK_ORDER


HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
EMBED_DIR = REPORTS / "embeddings" / "embeddings"
OUT = REPORTS / "embeddings"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------- I/O ----------------

def load_task_embeddings(task: str):
    p = EMBED_DIR / f"{task}.npz"
    if not p.exists():
        return None, None
    z = np.load(p, allow_pickle=True)
    return z["indices"], z["vectors"].astype(np.float32)


def load_kendall_matrix():
    rows = list(csv.DictReader(open(REPORTS / "correlation" / "pairwise_kendall.csv")))
    tasks = [r["task"] for r in rows]
    M = np.zeros((len(tasks), len(tasks)))
    for i, r in enumerate(rows):
        for j, t in enumerate(tasks):
            v = r[t]
            M[i, j] = float(v) if v else np.nan
    return tasks, M


# ---------------- centroid similarity ----------------

def compute_centroids(tasks):
    """Return (tasks_present, centroids) — L2-normalised mean per task."""
    centroids = []
    present = []
    for t in tasks:
        idx, vecs = load_task_embeddings(t)
        if vecs is None:
            print(f"  centroid: skip {t} (no cache)")
            continue
        c = vecs.mean(axis=0)
        n = float(np.linalg.norm(c))
        if n > 0:
            c /= n
        centroids.append(c)
        present.append(t)
    return present, np.stack(centroids) if centroids else np.zeros((0, 0))


def cosine_matrix(C):
    return C @ C.T


# ---------------- EVoC + purity ----------------

def concat_all_embeddings(tasks):
    parts_vecs = []
    parts_labels = []
    parts_idx = []
    for t in tasks:
        idx, vecs = load_task_embeddings(t)
        if vecs is None:
            continue
        parts_vecs.append(vecs)
        parts_labels.extend([t] * len(vecs))
        parts_idx.extend(list(idx))
    if not parts_vecs:
        return None, None, None
    X = np.vstack(parts_vecs)
    return X, np.asarray(parts_labels), np.asarray(parts_idx, dtype=object)


def cluster_purity(labels, source_tasks):
    """Per-cluster: dominant task, purity, size, top-3 task shares."""
    rows = []
    uniq = sorted(set(labels.tolist()))
    for c in uniq:
        if c < 0:
            continue
        mask = labels == c
        size = int(mask.sum())
        ctr = Counter(source_tasks[mask].tolist())
        dom_task, dom_n = ctr.most_common(1)[0]
        purity = dom_n / size
        top3 = [(t, n / size) for t, n in ctr.most_common(3)]
        rows.append({
            "cluster": int(c),
            "size": size,
            "dominant_task": dom_task,
            "purity": round(purity, 4),
            "top_3": ";".join(f"{t}={s:.3f}" for t, s in top3),
            "n_distinct_tasks": len(ctr),
        })
    rows.sort(key=lambda r: -r["size"])
    return rows


def task_overlap_matrix(labels, source_tasks, present, threshold_share=0.10):
    """For each pair (a, b): fraction of a's questions in clusters that
    also contain >= threshold_share of b's questions."""
    n_per_task = Counter(source_tasks.tolist())
    cluster_task_counts = {}  # cluster_id -> Counter
    for lbl, t in zip(labels.tolist(), source_tasks.tolist()):
        if lbl < 0:
            continue
        cluster_task_counts.setdefault(lbl, Counter())[t] += 1

    n = len(present)
    M = np.zeros((n, n))
    for a_i, a in enumerate(present):
        a_in_share = {}  # cluster -> count of a's questions in it
        # gather counts of a per cluster
        for cl, ctr in cluster_task_counts.items():
            if ctr.get(a, 0) > 0:
                a_in_share[cl] = ctr[a]
        for b_i, b in enumerate(present):
            if a_i == b_i:
                M[a_i, b_i] = 1.0
                continue
            shared = 0
            for cl, a_n in a_in_share.items():
                ctr = cluster_task_counts[cl]
                b_n = ctr.get(b, 0)
                if b_n / max(1, n_per_task[b]) >= threshold_share:
                    shared += a_n
            M[a_i, b_i] = shared / max(1, n_per_task[a])
    return M


# ---------------- Mantel-style test ----------------

def mantel(A, B, n_perm=999, seed=0):
    """Return (observed_pearson_correlation, two-sided p-value).

    Both A and B are square symmetric matrices. We compare upper-triangles.
    """
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    a = A[iu]
    b = B[iu]
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan"), float("nan")
    obs = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    geq = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        Bp = B[np.ix_(perm, perm)]
        bp = Bp[iu][mask]
        r = np.corrcoef(a, bp)[0, 1]
        if abs(r) >= abs(obs):
            geq += 1
    return obs, (geq + 1) / (n_perm + 1)


# ---------------- main ----------------

def write_matrix_csv(path, M, tasks):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", *tasks])
        for i, t in enumerate(tasks):
            row = [t] + [
                f"{M[i, j]:.4f}" if not np.isnan(M[i, j]) else ""
                for j in range(M.shape[1])
            ]
            w.writerow(row)


def quadrant(sim, tau, sim_thr=0.85, tau_thr=0.5):
    s = sim >= sim_thr
    t = abs(tau) >= tau_thr
    if s and t:
        return "redundant"
    if (not s) and t:
        return "general-axis-only"
    if s and (not t):
        return "format-quirk"
    return "diversifying"


def main():
    print("=== centroid similarity ===")
    present, C = compute_centroids(TASK_ORDER)
    if C.size == 0:
        raise SystemExit("No embeddings cached. Run analysis.embed first.")
    S = cosine_matrix(C)
    write_matrix_csv(OUT / "centroid_similarity.csv", S, present)
    print(f"  centroids for {len(present)}/{len(TASK_ORDER)} tasks; matrix saved")

    # ---------- centroid vs Kendall ----------
    k_tasks, K = load_kendall_matrix()
    pos = {t: i for i, t in enumerate(k_tasks)}
    rows = []
    for i, a in enumerate(present):
        for j, b in enumerate(present):
            if j <= i:
                continue
            if a not in pos or b not in pos:
                continue
            tau = K[pos[a], pos[b]]
            sim = float(S[i, j])
            rows.append({
                "task_a": a, "task_b": b,
                "semantic_sim": round(sim, 4),
                "kendall_tau": round(float(tau), 4) if not np.isnan(tau) else "",
                "quadrant": quadrant(sim, float(tau)) if not np.isnan(tau) else "",
            })
    with open(OUT / "centroid_vs_kendall.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Mantel-style correlation between the two matrices (aligned on `present`)
    K_aligned = np.full((len(present), len(present)), np.nan)
    for i, a in enumerate(present):
        for j, b in enumerate(present):
            if a in pos and b in pos:
                K_aligned[i, j] = K[pos[a], pos[b]]
    mantel_r, mantel_p = mantel(S, K_aligned, n_perm=1000)
    (OUT / "mantel.json").write_text(json.dumps({
        "pearson_r": mantel_r,
        "p_value": mantel_p,
        "n_perm": 1000,
    }, indent=2))
    print(f"  Mantel r={mantel_r:.4f}  p={mantel_p:.4f}")

    # ---------- EVoC clustering over all questions ----------
    print()
    print("=== EVoC clustering ===")
    X, src_tasks, src_idx = concat_all_embeddings(TASK_ORDER)
    if X is None:
        raise SystemExit("No embeddings; aborting clustering.")
    print(f"  X.shape = {X.shape}")
    # Use the persistence-selected layer (macro structure). We separately
    # compute per-question purity at the finest base layer for a granular
    # "is this content unique" signal.
    labels, clusterer = evoc_cluster(
        X, base_min_cluster_size=10, n_neighbors=15, min_samples=5,
        layer="persistent",
    )
    n_clusters = int(labels.max() + 1) if labels.size else 0
    n_noise = int((labels == -1).sum())
    print(f"  {n_clusters} clusters, {n_noise} noise points")

    with open(OUT / "evoc_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "index", "cluster"])
        for t, idx, c in zip(src_tasks.tolist(), src_idx.tolist(), labels.tolist()):
            w.writerow([t, idx, int(c)])

    # ---------- per-cluster purity (persistent layer) ----------
    purity_rows = cluster_purity(labels, src_tasks)
    with open(OUT / "cluster_task_purity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(purity_rows[0].keys()))
        w.writeheader()
        w.writerows(purity_rows)
    print(f"  cluster purity table: {len(purity_rows)} clusters (persistent layer)")

    # ---------- finer base-layer purity (for the per-question content-uniqueness signal) ----------
    base_labels = np.asarray(clusterer.cluster_layers_[0], dtype=np.int32)
    base_purity = cluster_purity(base_labels, src_tasks)
    with open(OUT / "cluster_task_purity_base.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(base_purity[0].keys()))
        w.writeheader()
        w.writerows(base_purity)
    print(f"  base-layer purity table: {len(base_purity)} clusters")

    # ---------- task overlap matrix ----------
    O = task_overlap_matrix(labels, src_tasks, present, threshold_share=0.10)
    write_matrix_csv(OUT / "task_overlap_matrix.csv", O, present)

    # ---------- ranked overlap pairs ----------
    overlap_rows = []
    for i, a in enumerate(present):
        for j, b in enumerate(present):
            if j <= i:
                continue
            ratio_ab = float(O[i, j])
            ratio_ba = float(O[j, i])
            mean = 0.5 * (ratio_ab + ratio_ba)
            overlap_rows.append({
                "task_a": a, "task_b": b,
                "overlap_a_in_b_clusters": round(ratio_ab, 4),
                "overlap_b_in_a_clusters": round(ratio_ba, 4),
                "mean_overlap": round(mean, 4),
            })
    overlap_rows.sort(key=lambda r: -r["mean_overlap"])
    md = ["# Cross-task content-overlap (EVoC clusters, ≥10 % share threshold)",
          "",
          "| Mean overlap | Task A | Task B | A→B | B→A |",
          "|---|---|---|---|---|"]
    for r in overlap_rows[:30]:
        md.append(
            f"| {r['mean_overlap']:.3f} | {r['task_a']} | {r['task_b']} | "
            f"{r['overlap_a_in_b_clusters']:.3f} | {r['overlap_b_in_a_clusters']:.3f} |"
        )
    (OUT / "overlap_pairs.md").write_text("\n".join(md))

    print(f"\nwrote artifacts to {OUT}/")


if __name__ == "__main__":
    main()
