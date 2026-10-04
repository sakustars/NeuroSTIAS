"""Remaining V4.4.1 analyses, ported: overview, NNLS deconvolution, graph-convolution
embedding, metabolic panels, colocalisation candidates and the virtual-intervention sandbox.

Implementation-level notes carried over from V4.4.1 (``METHOD_PROVENANCE``):

* deconvolution: local non-negative least squares on a marker basis or a user
  reference; no external API, no runtime download;
* graph convolution: an own deterministic two-layer symmetric-normalised
  propagation baseline, not an external GNN package;
* metabolic panels: z-score panel activity; exploratory pathway activity, not flux;
* colocalisation: Moran's I plus local correlation as a heuristic candidate
  screen, not a formal bivariate spatial significance test;
* virtual intervention: a heuristic network/spatial perturbation sandbox; it
  must not be interpreted as an experimental knockout or causal effect.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..atlas import markers
from ..core import graph
from ..core.io import detect_species, gene_symbols
from ..spatial3d.svg import morans_i

METABOLIC_PANELS = {
    "Glycolysis": ("HK1", "HK2", "GPI", "PFKP", "ALDOA", "GAPDH", "PGK1", "ENO1", "PKM", "LDHA"),
    "Oxidative phosphorylation": ("NDUFS1", "NDUFS2", "SDHA", "UQCRC1", "COX4I1", "ATP5F1A", "ATP5F1B"),
    "TCA cycle": ("CS", "ACO2", "IDH3A", "OGDH", "SUCLG1", "SDHA", "FH", "MDH2"),
    "Fatty acid oxidation": ("CPT1A", "CPT2", "ACADM", "ACADVL", "HADHA", "HADHB", "ECHS1"),
    "Pentose phosphate": ("G6PD", "PGLS", "PGD", "RPIA", "RPE", "TKT", "TALDO1"),
    # brain-specific metabolic coupling
    "Astrocyte-neuron lactate shuttle": ("SLC16A1", "SLC16A3", "SLC16A7", "LDHA", "LDHB", "PFKFB3"),
    "Glutamate-glutamine cycle": ("GLUL", "GLS", "SLC1A2", "SLC1A3", "SLC38A1", "SLC38A3"),
    "Cholesterol synthesis (myelin)": ("HMGCR", "HMGCS1", "SQLE", "DHCR24", "FDFT1", "LSS", "CYP51A1"),
}


def _X(adata, cols=None):
    X = adata.layers["log_normalized"] if "log_normalized" in adata.layers else adata.X
    if cols is not None:
        X = X[:, cols]
    return X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)


def _coords(adata):
    c = np.asarray(adata.obsm["spatial_3d"] if "spatial_3d" in adata.obsm else adata.obsm["spatial"], dtype=float)
    return np.column_stack([c, np.zeros(len(c))]) if c.shape[1] == 2 else c


def _lookup(adata, species):
    syms = gene_symbols(adata).astype(str)
    return {g.upper(): i for i, g in enumerate(syms)}


def deconvolve_nnls(adata, reference: pd.DataFrame | None = None, species: str = "auto") -> pd.DataFrame:
    """Proportions per observation by NNLS against a reference (genes × types) or a binary marker basis."""
    species = detect_species(gene_symbols(adata)) if species == "auto" else species
    lk = _lookup(adata, species)
    if reference is not None:
        ref = reference.copy()
        ref.index = ref.index.astype(str).str.upper()
        common = [g for g in ref.index if g in lk]
        if len(common) < 10:
            raise ValueError("fewer than 10 reference genes overlap the data")
        B = ref.loc[common].to_numpy(float)
        names = list(ref.columns)
        Y = _X(adata, [lk[g] for g in common])
    else:
        pnl = {k: [g.upper() for g in v if g.upper() in lk] for k, v in markers.panels("classes", species).items()}
        pnl = {k: v for k, v in pnl.items() if len(v) >= 2}
        genes = sorted({g for v in pnl.values() for g in v})
        if len(pnl) < 2:
            raise ValueError("not enough marker panels overlap the data; no deconvolution produced")
        B = np.zeros((len(genes), len(pnl)))
        pos = {g: i for i, g in enumerate(genes)}
        for j, v in enumerate(pnl.values()):
            B[[pos[g] for g in v], j] = 1.0
        names = list(pnl)
        Y = _X(adata, [lk[g] for g in genes])
    P = np.zeros((adata.n_obs, B.shape[1]))
    for i, y in enumerate(np.maximum(Y, 0)):
        w, _ = nnls(B, y)
        P[i] = w / w.sum() if w.sum() > 0 else 0
    return pd.DataFrame(P, index=adata.obs_names, columns=names)


def gcn_embedding(adata, k: int = 12, n_components: int = 16, n_clusters: int = 6, z_scale: float = 1.0,
                  block_key: str | None = None, seed: int = 42) -> np.ndarray:
    X = _X(adata)
    feats = PCA(n_components=min(n_components, X.shape[0] - 1, X.shape[1]), random_state=seed).fit_transform(X)
    blocks, _ = graph.block_labels(adata.obs, block_key)
    A = graph.knn_graph(_coords(adata), blocks, k=k, z_scale=z_scale) + sparse.eye(adata.n_obs)
    d = np.asarray(A.sum(1)).ravel()
    Dn = sparse.diags(1 / np.sqrt(np.maximum(d, 1e-12)))
    Nrm = Dn @ A @ Dn
    emb = np.asarray(Nrm @ np.maximum(Nrm @ feats, 0))
    adata.obsm["X_spatial_gcn"] = emb
    adata.obs["spatial_gcn_cluster"] = pd.Categorical(
        [f"GCN_{x + 1}" for x in KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(emb)])
    return emb


def metabolic_scores(adata, group_key: str | None = None, min_genes: int = 2) -> dict:
    lk = _lookup(adata, None)
    cov = pd.DataFrame([{"pathway": k, "panel_size": len(v), "matched": sum(g in lk for g in v)}
                        for k, v in METABOLIC_PANELS.items()])
    usable = {k: [lk[g] for g in v if g in lk] for k, v in METABOLIC_PANELS.items()}
    usable = {k: v for k, v in usable.items() if len(v) >= min_genes}
    if not usable:
        return {"status": "not_applicable", "reason": "fewer than 2 panel genes per pathway", "coverage": cov}
    cols = sorted({i for v in usable.values() for i in v})
    Z = _X(adata, cols)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 1e-12, Z.std(0), 1)
    pos = {c: j for j, c in enumerate(cols)}
    S = pd.DataFrame({k: Z[:, [pos[i] for i in v]].mean(1) for k, v in usable.items()}, index=adata.obs_names)
    out = {"status": "completed", "scores": S, "coverage": cov}
    if group_key:
        out["by_group"] = S.groupby(adata.obs[group_key].astype(str).to_numpy()).mean()
    return out


def colocalization_candidates(adata, n_top_genes: int = 30, k: int = 5, threshold: float = 0.2,
                              block_key: str | None = None) -> dict:
    X = adata.layers["log_normalized"] if "log_normalized" in adata.layers else adata.X
    var = np.asarray(X.power(2).mean(0) - np.square(X.mean(0))).ravel() if sparse.issparse(X) else np.var(np.asarray(X), 0)
    idx = np.argsort(var)[::-1][:n_top_genes]
    E = _X(adata, idx)
    blocks, _ = graph.block_labels(adata.obs, block_key)
    W = graph.knn_graph(_coords(adata), blocks, k=k)
    I = morans_i(E, W)
    S = graph.row_normalize(W) @ E
    Sc = (S - S.mean(0)) / np.where(S.std(0) > 0, S.std(0), 1)
    C = (Sc.T @ Sc) / len(Sc)
    names = adata.var_names[idx].astype(str)
    pairs = [{"gene_a": names[i], "gene_b": names[j], "local_corr": float(C[i, j])}
             for i in range(len(idx)) for j in range(i + 1, len(idx)) if abs(C[i, j]) >= threshold]
    return {"morans_i": pd.DataFrame({"gene": names, "morans_i": I}).sort_values("morans_i", ascending=False),
            "pairs": pd.DataFrame(pairs), "is_significance_test": False}


def virtual_intervention(adata, n_top_genes: int = 50, corr_threshold: float = 0.2, top_n_remove: int = 10,
                         block_key: str | None = None) -> dict:
    """Heuristic sandbox: remove hub genes from a spatially smoothed co-expression network and
    report the drop in global efficiency. Exploratory only; not a causal knockout."""
    X = adata.layers["log_normalized"] if "log_normalized" in adata.layers else adata.X
    var = np.asarray(X.power(2).mean(0) - np.square(X.mean(0))).ravel() if sparse.issparse(X) else np.var(np.asarray(X), 0)
    idx = np.argsort(var)[::-1][:n_top_genes]
    E = _X(adata, idx)
    blocks, _ = graph.block_labels(adata.obs, block_key)
    S = graph.row_normalize(graph.knn_graph(_coords(adata), blocks, k=6)) @ E
    C = np.corrcoef(S.T)
    names = adata.var_names[idx].astype(str)
    G = nx.Graph()
    G.add_nodes_from(names)
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if abs(C[i, j]) >= corr_threshold:
                G.add_edge(names[i], names[j], weight=abs(C[i, j]))
    if G.number_of_edges() == 0:
        return {"status": "unavailable", "reason": "no spatial co-expression edge passed corr_threshold"}
    base = nx.global_efficiency(G)
    hubs = [n for n, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_n_remove]]
    impact = {}
    for h in hubs:
        H = G.copy()
        H.remove_node(h)
        impact[h] = float(base - nx.global_efficiency(H))
    return {"status": "exploratory", "baseline_efficiency": float(base), "n_edges": G.number_of_edges(),
            "efficiency_drop_by_hub": impact,
            "interpretation": "heuristic network sensitivity; not an experimental or causal perturbation"}
