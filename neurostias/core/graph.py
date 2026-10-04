"""Spatial neighbour graphs.

Two rules apply throughout:

1. **No edges between independent samples.** Different animals/donors share
   coordinate ranges but are not physically adjacent; connecting them creates
   false neighbours. (Ported from 3D-STIAS V4.4.1 ``_within_sample_knn``.)
2. **Anisotropy is explicit.** In serial-section data the distance between
   sections (z) is measured differently from in-plane spot spacing (x, y) and is
   often much larger. ``z_scale`` multiplies z before neighbour search, so
   ``z_scale=0`` gives independent 2D graphs per section, ``z_scale=1`` treats
   physical µm isotropically, and other values are tuned on held-out sections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

BLOCK_KEYS = ("replicate", "sample_id", "sample", "donor", "library_id", "library", "batch")


def block_labels(obs: pd.DataFrame, key: str | None = None) -> tuple[np.ndarray, str | None]:
    """Labels of independent samples (no edges allowed between blocks)."""
    if key is not None:
        return obs[key].astype(str).to_numpy(), key
    for k in BLOCK_KEYS:
        if k in obs.columns:
            labels = obs[k].astype("string").fillna("Unknown").to_numpy(dtype=str)
            if len(np.unique(labels)) > 1:
                return labels, k
    return np.repeat("sample_0", len(obs)), None


def knn_lists(coords: np.ndarray, blocks: np.ndarray, k: int) -> list[list[int]]:
    """k nearest neighbours (excluding self) computed within each block."""
    coords = np.asarray(coords, dtype=float)
    out: list[list[int]] = [[] for _ in range(len(coords))]
    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        if len(idx) == 1:
            out[int(idx[0])] = [int(idx[0])]
            continue
        _, nn = cKDTree(coords[idx]).query(coords[idx], k=min(k + 1, len(idx)))
        nn = np.atleast_2d(nn)
        for row, g in enumerate(idx):
            mapped = [int(j) for j in idx[nn[row]] if j != g][:k]
            out[int(g)] = mapped or [int(g)]
    return out


def scaled_coords(coords: np.ndarray, z_scale: float = 1.0) -> np.ndarray:
    c = np.asarray(coords, dtype=float).copy()
    if c.shape[1] == 3:
        c[:, 2] *= z_scale
    return c


def knn_graph(coords: np.ndarray, blocks: np.ndarray, k: int = 6, z_scale: float = 1.0,
              sections: np.ndarray | None = None, symmetric: bool = True,
              weight: str = "binary") -> sparse.csr_matrix:
    """Sparse kNN adjacency (n x n) built within blocks on anisotropy-scaled coordinates.

    If ``z_scale == 0`` and ``sections`` is given, neighbours are restricted to the
    same section (pure 2D), even if sections overlap in x/y.
    """
    c = scaled_coords(coords, z_scale)
    groups = blocks
    if sections is not None and z_scale == 0:
        groups = np.char.add(np.char.add(blocks.astype(str), "|"), np.asarray(sections).astype(str))
    lists = knn_lists(c, groups, k)
    rows = np.repeat(np.arange(len(lists)), [len(l) for l in lists])
    cols = np.fromiter((j for l in lists for j in l), dtype=int, count=len(rows))
    if weight == "gaussian":
        d = np.linalg.norm(c[rows] - c[cols], axis=1)
        bw = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        vals = np.exp(-(d / bw) ** 2)
    else:
        vals = np.ones(len(rows))
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lists), len(lists)))
    W.setdiag(0)
    W.eliminate_zeros()
    if symmetric:
        W = W.maximum(W.T)
    return W.tocsr()


def cross_section_knn(coords: np.ndarray, sections: np.ndarray, blocks: np.ndarray,
                      k_within: int = 6, k_between: int = 3, z_scale: float = 1.0) -> sparse.csr_matrix:
    """STAGATE-3D-style graph: kNN inside each section plus kNN to adjacent sections.

    Unlike a single isotropic kNN, every spot is guaranteed ``k_between``
    neighbours in each adjacent section of the same block, and inter-section edge
    weights decay with the anisotropy-scaled 3D distance relative to in-plane spacing.
    """
    coords = np.asarray(coords, dtype=float)
    sections = np.asarray(sections).astype(str)
    n = len(coords)
    W_in = knn_graph(coords, blocks, k=k_within, z_scale=0.0, sections=sections)
    rows, cols, vals = [], [], []
    xy = coords[:, :2]
    inplane = []
    for b in np.unique(blocks):
        bidx = np.flatnonzero(blocks == b)
        sec_z = (pd.DataFrame({"s": sections[bidx], "z": coords[bidx, 2]})
                 .groupby("s")["z"].median().sort_values())
        order = list(sec_z.index)
        trees = {s: cKDTree(xy[bidx[sections[bidx] == s]]) for s in order}
        for s in order:
            si = bidx[sections[bidx] == s]
            if len(si) > 1:
                inplane.append(np.median(trees[s].query(xy[si], k=2)[0][:, 1]))
        for a, bsec in zip(order[:-1], order[1:]):
            ia = bidx[sections[bidx] == a]
            ib = bidx[sections[bidx] == bsec]
            dz = abs(sec_z[bsec] - sec_z[a]) * z_scale
            for src, dst_idx, tree in ((ia, ib, trees[bsec]), (ib, ia, trees[a])):
                kk = min(k_between, len(dst_idx))
                d, nn = tree.query(xy[src], k=kk)
                d = np.atleast_2d(d.T).T if kk == 1 else d
                nn = np.atleast_2d(nn.T).T if kk == 1 else nn
                d3 = np.sqrt(d ** 2 + dz ** 2)
                rows.append(np.repeat(src, kk))
                cols.append(dst_idx[nn.ravel()])
                vals.append(d3.ravel())
    if not rows:
        return W_in
    rows = np.concatenate(rows); cols = np.concatenate(cols); d3 = np.concatenate(vals)
    spacing = float(np.median(inplane)) if inplane else 1.0
    w = np.exp(-(d3 / max(spacing, 1e-9)) ** 2 / 2.0)
    W_between = sparse.csr_matrix((w, (rows, cols)), shape=(n, n))
    W_between = W_between.maximum(W_between.T)
    return (W_in + W_between).tocsr()


def row_normalize(W: sparse.spmatrix) -> sparse.csr_matrix:
    d = np.asarray(W.sum(axis=1)).ravel()
    d[d == 0] = 1.0
    return sparse.diags(1.0 / d) @ W


def neighbourhood_mean(X, W: sparse.spmatrix):
    """Mean of features over graph neighbours (rows of W normalised)."""
    return row_normalize(W) @ X
