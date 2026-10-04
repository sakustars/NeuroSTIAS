"""Differential expression.

``region_de_v441`` reproduces 3D-STIAS V4.4.1 exactly as it behaved: spatial
regions from k-means on coordinates, then an observation-level Welch t-test per
gene between every pair of regions with BH correction. It is kept for
regression and as the baseline in experiment E05, because it treats spots as
independent replicates and re-uses the same data to define and test regions.

``replicate_de`` is the NeuroSTIAS default: pseudobulk by biological replicate
(``core.stats.pseudobulk_test``), with no p-values when fewer than two replicates
exist per group.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.cluster import KMeans

from ..core.stats import bh, pseudobulk_test


def _dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)


def region_de_v441(adata, n_regions: int = 8, min_cells: int = 10, padj_threshold: float = 0.05,
                   lfc_threshold: float = 1.0, region_method: str = "spatial", seed: int = 42,
                   region_key: str | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """V4.4.1 behaviour. Returns (summary table, {comparison: per-gene table})."""
    if region_key is not None:
        regions = adata.obs[region_key].astype(str).to_numpy()
    elif region_method == "spatial":
        coords = np.asarray(adata.obsm["spatial_3d"] if "spatial_3d" in adata.obsm else adata.obsm["spatial"])
        regions = KMeans(n_clusters=n_regions, random_state=seed, n_init=10).fit_predict(coords).astype(str)
    else:
        regions = KMeans(n_clusters=n_regions, random_state=seed, n_init=10).fit_predict(adata.obsm["X_pca"]).astype(str)
    adata.obs["spatial_region"] = pd.Categorical(regions)
    X = _dense(adata.layers["log_normalized"] if "log_normalized" in adata.layers else adata.X)
    genes = adata.var_names.astype(str)
    out, summary = {}, []
    for a, b in itertools.combinations(sorted(np.unique(regions)), 2):
        ma, mb = regions == a, regions == b
        if ma.sum() <= min_cells or mb.sum() <= min_cells:
            continue
        xa, xb = X[ma], X[mb]
        # V4.4.1 tests only genes variable in both regions and expressed in at least one
        ok = (xa.std(0) > 0) & (xb.std(0) > 0) & (xa.mean(0) + xb.mean(0) > 0)
        _, p = stats.ttest_ind(xa[:, ok], xb[:, ok], axis=0, equal_var=False)
        keep = np.isfinite(p)
        p = np.clip(p[keep], np.finfo(float).tiny, 1.0)
        lfc = ((xa[:, ok].mean(0) - xb[:, ok].mean(0)) / np.log(2))[keep]  # ln-space difference / ln 2, as in V4.4.1
        df = pd.DataFrame({"gene": genes[ok][keep], "log2fc": lfc, "pval": p})
        df["padj"] = bh(df["pval"])
        df["significant"] = (df["padj"] < padj_threshold) & (df["log2fc"].abs() > lfc_threshold)
        name = f"Region_{a}_vs_Region_{b}"
        out[name] = df
        summary.append({"comparison": name, "n_a": int(ma.sum()), "n_b": int(mb.sum()),
                        "n_significant": int(df["significant"].sum())})
    return pd.DataFrame(summary), out


def wilcoxon_de(adata, group_key: str, group_a: str, group_b: str) -> pd.DataFrame:
    """Scanpy default observation-level Wilcoxon test (baseline for E05)."""
    import scanpy as sc
    sub = adata[adata.obs[group_key].astype(str).isin([group_a, group_b])].copy()
    sub.obs["_g"] = sub.obs[group_key].astype(str).astype("category")
    layer = "log_normalized" if "log_normalized" in sub.layers else None
    sc.tl.rank_genes_groups(sub, "_g", groups=[group_a], reference=group_b, method="wilcoxon", layer=layer)
    r = sc.get.rank_genes_groups_df(sub, group=group_a)
    return r.rename(columns={"names": "gene", "pvals": "pval", "pvals_adj": "padj", "logfoldchanges": "log2fc"})


def replicate_de(adata, group_key: str, group_a: str, group_b: str, replicate_key: str,
                 min_cells: int = 10, paired: bool = True) -> pd.DataFrame:
    from ..core.io import counts_matrix
    X = counts_matrix(adata)
    return pseudobulk_test(X, adata.var_names.astype(str), adata.obs[group_key].astype(str).reset_index(drop=True),
                           adata.obs[replicate_key].astype(str).reset_index(drop=True), group_a, group_b,
                           min_cells=min_cells, paired=paired)
