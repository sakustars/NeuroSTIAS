"""Spatial niches: clusters of local cell-type composition (ported from V4.4.1 ``spatial_niche_analysis_3d``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..core import graph


def niches(adata, label_key: str, n_niches: int = 6, k: int = 15, z_scale: float = 1.0,
           block_key: str | None = None, seed: int = 0, key_added: str = "spatial_niche") -> pd.DataFrame:
    labels = adata.obs[label_key].astype(str).to_numpy()
    cats, enc = np.unique(labels, return_inverse=True)
    coords = np.asarray(adata.obsm["spatial_3d"] if "spatial_3d" in adata.obsm else adata.obsm["spatial"], dtype=float)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])
    blocks, _ = graph.block_labels(adata.obs, block_key)
    nn = graph.knn_lists(graph.scaled_coords(coords, z_scale), blocks, k)
    comp = np.zeros((adata.n_obs, len(cats)))
    for i, ids in enumerate(nn):
        comp[i] = np.bincount(enc[ids], minlength=len(cats)) / len(ids)
    n = int(min(max(2, n_niches), len(np.unique(comp, axis=0))))
    lab = KMeans(n_clusters=n, n_init=20, random_state=seed).fit_predict(comp)
    adata.obs[key_added] = pd.Categorical([f"Niche_{x + 1}" for x in lab])
    adata.obsm[f"{key_added}_composition"] = comp
    table = pd.DataFrame(comp, index=adata.obs_names, columns=cats)
    table[key_added] = adata.obs[key_added].astype(str).to_numpy()
    return table
