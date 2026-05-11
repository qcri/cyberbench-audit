"""Thin wrapper around evoc.EVoC for benchmark question clustering."""

from __future__ import annotations

from typing import Optional

import numpy as np


def cluster(
    X: np.ndarray,
    base_min_cluster_size: int = 10,
    n_neighbors: int = 15,
    min_samples: int = 5,
    random_state: int = 0,
    layer: str = "base",
):
    """Return integer cluster labels (-1 for noise) using EVoC.

    `layer`:
      - "base"        : finest-grain (cluster_layers_[0])
      - "persistent"  : the layer EVoC's fit_predict picks by max persistence
      - int           : explicit index into cluster_layers_

    Lazy-imports evoc so the lib loads even without the dep.
    """
    import evoc
    clusterer = evoc.EVoC(
        base_min_cluster_size=base_min_cluster_size,
        n_neighbors=n_neighbors,
        min_samples=min_samples,
        random_state=random_state,
    )
    persistent_labels = clusterer.fit_predict(X)
    if layer == "persistent":
        labels = persistent_labels
    elif layer == "base":
        labels = clusterer.cluster_layers_[0]
    elif isinstance(layer, int):
        labels = clusterer.cluster_layers_[layer]
    else:
        labels = persistent_labels
    return np.asarray(labels, dtype=np.int32), clusterer
