"""Pseudotime by diffusion pseudotime (ported from V4.4.1 ``cell_trajectory_inference_3d``).

Root selection, in priority order: an explicit index; the diffusion-space medoid
of a root label (e.g. ``Radial glia / NPC`` from ``neurodev_stage``); the
observation with the highest (or lowest) expression of a root gene; otherwise a
graph-diameter endpoint of the largest connected component, which gives a stable
but *not biological* direction and is flagged as such. Sensitivity is assessed
by re-running shortest-path pseudotime over perturbed kNN sizes and nearby root
candidates; the reported spread is not a statistical confidence interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import spearmanr
from sklearn.neighbors import kneighbors_graph


def _knn(emb, k):
    g = kneighbors_graph(emb, n_neighbors=min(max(2, k), len(emb) - 1), mode="distance", include_self=False).tocsr()
    g.data = np.maximum(g.data, np.finfo(float).eps)
    g = g.maximum(g.T).tocsr()
    g.setdiag(0)
    g.eliminate_zeros()
    return g


def _norm(v):
    v = np.asarray(v, dtype=float)
    f = np.isfinite(v)
    if f.any() and np.ptp(v[f]) > 0:
        v[f] = (v[f] - v[f].min()) / np.ptp(v[f])
    return v


def pseudotime(adata, n_pcs: int = 30, n_neighbors: int = 15, root_index: int | None = None,
               root_label_key: str | None = None, root_label: str | None = None, root_gene: str | None = None,
               root_gene_mode: str = "high", sensitivity_repeats: int = 6, seed: int = 0) -> dict:
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1, adata.n_obs - 1), random_state=seed)
    npc = min(n_pcs, adata.obsm["X_pca"].shape[1])
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=npc, random_state=seed, key_added="traj")
    sc.tl.diffmap(adata, n_comps=4, neighbors_key="traj")
    dm = np.asarray(adata.obsm["X_diffmap"])[:, 1:4]
    G = _knn(dm, n_neighbors)
    if root_index is not None:
        root, info = int(root_index), {"strategy": "explicit_index", "biological_direction": True}
    elif root_label_key and root_label:
        cand = np.flatnonzero(adata.obs[root_label_key].astype(str).to_numpy() == root_label)
        if not len(cand):
            raise ValueError(f"no observations labelled {root_label!r} in {root_label_key}")
        root = int(cand[np.argmin(np.linalg.norm(dm[cand] - np.median(dm[cand], 0), axis=1))])
        info = {"strategy": "root_label_medoid", "value": root_label, "biological_direction": True}
    elif root_gene:
        v = adata[:, root_gene].X
        v = np.asarray(v.toarray() if hasattr(v, "toarray") else v).ravel()
        root = int(np.nanargmin(v) if root_gene_mode == "low" else np.nanargmax(v))
        info = {"strategy": "root_gene", "value": root_gene, "mode": root_gene_mode, "biological_direction": True}
    else:
        n_comp, comp = connected_components(G, directed=False)
        nodes = np.flatnonzero(comp == np.argmax(np.bincount(comp)))
        a = int(nodes[np.argmax(dijkstra(G, directed=False, indices=int(nodes[0]))[nodes])])
        b = int(nodes[np.argmax(dijkstra(G, directed=False, indices=a)[nodes])])
        names = adata.obs_names.astype(str)
        root = a if names[a] <= names[b] else b
        info = {"strategy": "graph_diameter_endpoint", "biological_direction": False,
                "warning": "no biological root given; direction is computational only"}
    adata.uns["iroot"] = root
    sc.tl.dpt(adata, n_dcs=3, neighbors_key="traj")
    pt = _norm(adata.obs["dpt_pseudotime"].replace([np.inf], np.nan).to_numpy(copy=True))
    adata.obs["pseudotime"] = pt
    base = dijkstra(G, directed=False, indices=root)
    reach = np.flatnonzero(np.isfinite(base))
    roots = reach[np.argsort(base[reach])][:3]
    runs, rhos = [], []
    for r in range(sensitivity_repeats):
        k = max(2, int(round(n_neighbors * (0.5, 1.0, 1.5)[r % 3])))
        d = _norm(dijkstra(_knn(dm, k), directed=False, indices=int(roots[(r // 3) % len(roots)])))
        d[~np.isfinite(d)] = np.nan
        runs.append(d)
        m = np.isfinite(d) & np.isfinite(pt)
        if m.sum() > 3:
            rhos.append(float(spearmanr(pt[m], d[m])[0]))
    stack = np.vstack(runs)
    iqr = np.nanpercentile(stack, 75, 0) - np.nanpercentile(stack, 25, 0)
    adata.obs["pseudotime_sensitivity_iqr"] = iqr
    return {"root": root, "root_info": info, "n_unreachable": int(np.isnan(pt).sum()),
            "sensitivity_median_spearman": float(np.median(rhos)) if rhos else None,
            "fraction_unstable_iqr_gt_0.2": float(np.nanmean(iqr > 0.2))}
