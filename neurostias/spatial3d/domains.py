"""Spatial domain detection with anisotropy-aware 2D/3D neighbourhood augmentation.

Each observation is represented by its own expression embedding (PCA) concatenated
with the mean embedding of its spatial neighbours (as in BANKSY, Singhal et al.
2024). The difference here is the graph: neighbours come from
``core.graph`` with an explicit ``z_scale`` (0 = per-section 2D; >0 = 3D
across sections of the same sample), never across independent samples.
Clusters come from k-means (when the number of domains is given, as in
benchmarks) or Leiden, optionally followed by a neighbour majority-vote
refinement (as in SpaGCN's refinement step).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..core import graph


@dataclass
class DomainParams:
    n_domains: int | None = 7
    n_pcs: int = 30
    n_hvg: int = 3000
    k_neighbors: int = 6
    lam: float = 0.8           # weight of the neighbourhood component (BANKSY default for domains)
    hops: int = 2              # neighbourhood smoothing depth
    z_scale: float = 0.0       # 0 => 2D per section
    cross_section: bool = False  # use guaranteed adjacent-section neighbours (cross_section_knn)
    k_between: int = 3
    refine: bool = True
    resolution: float = 0.5    # Leiden resolution when n_domains is None
    cluster_method: str = "kmeans"  # "kmeans" | "gmm" (Gaussian mixture on 20 PCs of the features; unequal domain sizes)
    seed: int = 0


def expression_embedding(adata, n_pcs: int, n_hvg: int, seed: int) -> np.ndarray:
    """PCA of log-normalised HVGs (computed jointly over all sections)."""
    X = adata.layers["log_normalized"] if "log_normalized" in adata.layers else adata.X
    tmp = sc.AnnData(X=X.copy() if sparse.issparse(X) else np.asarray(X, dtype=np.float32))
    tmp.var_names = adata.var_names
    n_hvg = min(n_hvg, adata.n_vars)
    if n_hvg < adata.n_vars:
        try:
            sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg, flavor="seurat")
        except Exception:
            sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg, flavor="cell_ranger")
        tmp = tmp[:, tmp.var["highly_variable"]].copy()
    sc.pp.scale(tmp, max_value=10)
    n = int(min(n_pcs, tmp.n_vars - 1, tmp.n_obs - 1))
    return PCA(n_components=n, random_state=seed).fit_transform(np.asarray(tmp.X))


def build_graph(adata, p: DomainParams, section_key: str | None, block_key: str | None) -> sparse.csr_matrix:
    coords = np.asarray(adata.obsm["spatial_3d"] if "spatial_3d" in adata.obsm else adata.obsm["spatial"], dtype=float)
    blocks, _ = graph.block_labels(adata.obs, block_key)
    sections = adata.obs[section_key].astype(str).to_numpy() if section_key else None
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])
    if p.cross_section and sections is not None and p.z_scale > 0:
        return graph.cross_section_knn(coords, sections, blocks, k_within=p.k_neighbors,
                                       k_between=p.k_between, z_scale=p.z_scale)
    return graph.knn_graph(coords, blocks, k=p.k_neighbors, z_scale=p.z_scale, sections=sections)


def augmented_features(emb: np.ndarray, W: sparse.csr_matrix, lam: float, hops: int) -> np.ndarray:
    Wn = graph.row_normalize(W)
    H = emb
    for _ in range(max(1, hops)):
        H = Wn @ H
    own = emb / (np.linalg.norm(emb, axis=1, keepdims=True).mean() + 1e-12)
    nbr = H / (np.linalg.norm(H, axis=1, keepdims=True).mean() + 1e-12)
    return np.hstack([np.sqrt(1 - lam) * own, np.sqrt(lam) * nbr])


def refine_labels(labels: np.ndarray, W: sparse.csr_matrix) -> np.ndarray:
    """Replace a label by its neighbours' majority when more than half the neighbours disagree."""
    labels = np.asarray(labels)
    codes, uniq = pd.factorize(labels)
    W = W.tocsr()
    out = codes.copy()
    for i in range(W.shape[0]):
        nb = W.indices[W.indptr[i]:W.indptr[i + 1]]
        if len(nb) == 0:
            continue
        counts = np.bincount(codes[nb], minlength=len(uniq))
        if counts[codes[i]] < len(nb) / 2:
            out[i] = int(np.argmax(counts))
    return uniq[out]


def detect_domains(adata, params: DomainParams | None = None, section_key: str | None = None,
                   block_key: str | None = None, key_added: str = "neuro_domain") -> dict:
    p = params or DomainParams()
    emb = expression_embedding(adata, p.n_pcs, p.n_hvg, p.seed)
    W = build_graph(adata, p, section_key, block_key)
    feats = augmented_features(emb, W, p.lam, p.hops)
    if p.n_domains and p.cluster_method == "gmm":
        from sklearn.mixture import GaussianMixture
        z = PCA(n_components=min(20, feats.shape[1] - 1), random_state=p.seed).fit_transform(feats)
        labels = GaussianMixture(n_components=p.n_domains, covariance_type="full", n_init=3,
                                 random_state=p.seed).fit_predict(z)
    elif p.n_domains:
        labels = KMeans(n_clusters=p.n_domains, n_init=10, random_state=p.seed).fit_predict(feats)
    else:
        tmp = sc.AnnData(X=feats.astype(np.float32))
        sc.pp.neighbors(tmp, use_rep="X", n_neighbors=15, random_state=p.seed)
        sc.tl.leiden(tmp, resolution=p.resolution, random_state=p.seed, flavor="igraph", n_iterations=2, directed=False)
        labels = tmp.obs["leiden"].astype(int).to_numpy()
    if p.refine:
        labels = refine_labels(labels, W)
    adata.obs[key_added] = pd.Categorical([f"D{int(l)}" for l in labels])
    adata.obsm[f"X_{key_added}_features"] = feats
    return {"params": asdict(p), "n_domains_found": int(len(np.unique(labels))),
            "n_edges": int(W.nnz), "mean_degree": float(W.nnz / max(adata.n_obs, 1))}
