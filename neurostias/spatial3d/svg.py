"""Spatially variable genes by Moran's I on a within-sample 2D/3D kNN graph.

Ported from 3D-STIAS V4.4.1 ``spatial_variability_analysis_3d``: permutation
p-values shuffle expression only within each independent sample (block), and
BH-FDR is applied across genes. The computation is vectorised over genes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ..core import graph
from ..core.stats import bh


def morans_i(X: np.ndarray, W: sparse.csr_matrix) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    num = np.einsum("ij,ij->j", Xc, W @ Xc)
    den = np.einsum("ij,ij->j", Xc, Xc)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (X.shape[0] / W.sum()) * num / den


def spatially_variable_genes(adata, n_top_genes: int = 500, k: int = 8, z_scale: float = 1.0,
                             n_permutations: int = 199, block_key: str | None = None,
                             section_key: str | None = None, seed: int = 0) -> pd.DataFrame:
    Xs = adata.layers["log_normalized"] if "log_normalized" in adata.layers else adata.X
    if sparse.issparse(Xs):
        mean = np.asarray(Xs.mean(axis=0)).ravel()
        var = np.asarray(Xs.power(2).mean(axis=0)).ravel() - mean ** 2
    else:
        var = np.var(np.asarray(Xs), axis=0)
    idx = np.argsort(var)[::-1][: min(n_top_genes, adata.n_vars)]
    idx = idx[var[idx] > 0]
    X = Xs[:, idx]
    X = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
    coords = np.asarray(adata.obsm["spatial_3d"] if "spatial_3d" in adata.obsm else adata.obsm["spatial"], dtype=float)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])
    blocks, bkey = graph.block_labels(adata.obs, block_key)
    sections = adata.obs[section_key].astype(str).to_numpy() if section_key else None
    W = graph.knn_graph(coords, blocks, k=k, z_scale=z_scale, sections=sections)
    I = morans_i(X, W)
    rng = np.random.default_rng(seed)
    exceed = np.zeros(len(idx))
    block_idx = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    for _ in range(n_permutations):
        perm = np.arange(len(X))
        for bi in block_idx:
            perm[bi] = rng.permutation(bi)
        exceed += morans_i(X[perm], W) >= I
    p = (1 + exceed) / (n_permutations + 1)
    res = pd.DataFrame({"gene": adata.var_names[idx].astype(str), "morans_i": I, "pval": p})
    res["padj"] = bh(res["pval"])
    res.attrs["block_key"] = bkey
    return res.sort_values("morans_i", ascending=False).reset_index(drop=True)
