"""Standard preprocessing (ported from 3D-STIAS V4.4.1 ``run_data_preprocessing``).

Defaults match V4.4.1: min_genes=200, min_cells=3, mitochondrial threshold 20 %,
target_sum=1e4, 2,000 Seurat-flavour HVGs, scaling clipped at 10, 50 PCs,
15 neighbours, Leiden clustering. Layers kept: ``counts``, ``normalized``,
``log_normalized``, ``scaled`` (HVG-scaled values in ``X`` only if requested).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scanpy as sc
from scipy import sparse


@dataclass
class PreprocessParams:
    min_genes: int = 200
    min_cells: int = 3
    max_pct_mt: float = 20.0
    target_sum: float = 1e4
    n_top_genes: int = 2000
    n_pcs: int = 50
    n_neighbors: int = 15
    leiden_resolution: float = 1.0
    compute_umap: bool = True
    seed: int = 42


def qc_metrics(adata) -> None:
    names = adata.var_names.astype(str)
    adata.var["mt"] = names.str.upper().str.startswith("MT-")
    adata.var["ribo"] = names.str.upper().str.match(r"^RP[SL]\d")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], percent_top=None, inplace=True, log1p=False)
    adata.obs["n_genes"] = adata.obs["n_genes_by_counts"]
    adata.obs["pct_counts_mt"] = adata.obs.get("pct_counts_mt", 0.0)


def looks_like_counts(X) -> bool:
    sample = X[: min(500, X.shape[0])]
    vals = sample.data if sparse.issparse(sample) else np.asarray(sample).ravel()
    vals = vals[vals != 0][:20000]
    return vals.size > 0 and np.all(vals >= 0) and np.allclose(vals, np.round(vals))


def preprocess(adata, params: PreprocessParams | None = None, filter_cells: bool = True):
    """Run QC → filter → normalise → log1p → HVG → scale → PCA → kNN → Leiden (→ UMAP).

    Returns (adata, report). If ``X`` is not integer counts (already normalised data,
    as in the Human Brain 3D release), normalisation is skipped and recorded.
    """
    p = params or PreprocessParams()
    report: dict = {"params": asdict(p), "n_obs_in": int(adata.n_obs), "n_vars_in": int(adata.n_vars)}
    adata.var_names_make_unique()
    is_counts = looks_like_counts(adata.X)
    report["input_is_integer_counts"] = bool(is_counts)
    qc_metrics(adata)
    if filter_cells:
        sc.pp.filter_genes(adata, min_cells=p.min_cells)
        keep = adata.obs["n_genes"] >= (p.min_genes if is_counts else 0)
        if adata.var["mt"].any():
            keep &= adata.obs["pct_counts_mt"] <= p.max_pct_mt
        adata._inplace_subset_obs(keep.to_numpy())
    report["n_obs_after_qc"] = int(adata.n_obs)
    report["n_vars_after_qc"] = int(adata.n_vars)
    # Only label a layer "counts" when it really holds integer counts.
    adata.layers["counts" if is_counts else "input"] = adata.X.copy()
    if is_counts:
        sc.pp.normalize_total(adata, target_sum=p.target_sum)
        adata.layers["normalized"] = adata.X.copy()
        sc.pp.log1p(adata)
    else:
        report["note"] = "input not integer counts; treated as already normalised (log1p applied only if max > 50)"
        mx = adata.X.max() if not sparse.issparse(adata.X) else adata.X.max()
        if float(mx) > 50:
            sc.pp.log1p(adata)
    adata.layers["log_normalized"] = adata.X.copy()
    n_hvg = int(min(p.n_top_genes, max(50, adata.n_vars - 1)))
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
    except Exception:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="cell_ranger")
    hv = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(hv, max_value=10)
    n_pcs = int(max(2, min(p.n_pcs, hv.n_vars - 1, hv.n_obs - 1)))
    sc.tl.pca(hv, n_comps=n_pcs, random_state=p.seed)
    adata.obsm["X_pca"] = hv.obsm["X_pca"]
    adata.uns["pca"] = hv.uns["pca"]
    sc.pp.neighbors(adata, n_neighbors=p.n_neighbors, n_pcs=n_pcs, random_state=p.seed)
    sc.tl.leiden(adata, resolution=p.leiden_resolution, random_state=p.seed,
                 flavor="igraph", n_iterations=2, directed=False)
    if p.compute_umap:
        sc.tl.umap(adata, random_state=p.seed)
    report.update({"n_hvg": int(adata.var["highly_variable"].sum()), "n_pcs": n_pcs,
                   "n_leiden_clusters": int(adata.obs["leiden"].nunique())})
    return adata, report
