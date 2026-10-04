"""V4.4.1 spatial community analysis (ported as-is for regression).

V4.4.1 chose between k-means on 3D coordinates ("spatial") and k-means on
standardised coordinates + 10 PCs ("combined"), with the cluster number from an
elbow heuristic. Coordinate-only k-means partitions space into compact blobs
regardless of expression; NeuroSTIAS uses ``spatial3d.domains`` instead and keeps
this for comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def elbow_k(coords: np.ndarray, n_obs: int, seed: int = 42) -> int:
    k = 2 if n_obs < 100 else 3 if n_obs < 500 else 4 if n_obs < 1000 else 6 if n_obs < 2000 else 8
    ks = range(2, min(10, n_obs // 10))
    inertia = [KMeans(n_clusters=kk, random_state=seed, n_init=10).fit(coords).inertia_ for kk in ks]
    if len(inertia) > 2:
        d = np.diff(inertia)
        k = int(np.argmax(d[:-1] / d[1:]) + 2)
    return k


def community_v441(adata, method: str = "spatial", n_clusters: int | None = None, seed: int = 42) -> np.ndarray:
    coords = np.asarray(adata.obsm["spatial_3d"] if "spatial_3d" in adata.obsm else adata.obsm["spatial"], dtype=float)
    k = n_clusters or elbow_k(coords, adata.n_obs, seed)
    if method == "combined" and "X_pca" in adata.obsm:
        feats = np.hstack([StandardScaler().fit_transform(coords),
                           StandardScaler().fit_transform(adata.obsm["X_pca"][:, :10])])
    else:
        feats = coords
    lab = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(feats)
    adata.obs["spatial_community"] = pd.Categorical([f"C{x}" for x in lab])
    return lab
