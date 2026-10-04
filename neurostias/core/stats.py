"""Replicate-aware statistics.

Spatial transcriptomics spots or cells from the same animal or donor are not
independent. Treating them as replicates (the default in most spot-level
differential-expression tools) inflates false positives. The functions here
aggregate to the biological unit first (pseudobulk) and test across units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, stat=np.mean):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(values, values.size, replace=True)) for _ in range(n_boot)])
    return (float(stat(values)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def bh(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def pseudobulk(X, groups: pd.Series, units: pd.Series, min_cells: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum counts per (unit, group). Returns (matrix units×genes, meta)."""
    key = units.astype(str) + "||" + groups.astype(str)
    codes, uniques = pd.factorize(key)
    n = len(uniques)
    indicator = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(n, len(codes)))
    summed = indicator @ (X if sparse.issparse(X) else sparse.csr_matrix(X))
    counts = np.bincount(codes, minlength=n)
    meta = pd.DataFrame({"unit": [u.split("||")[0] for u in uniques],
                         "group": [u.split("||")[1] for u in uniques],
                         "n_cells": counts})
    keep = counts >= min_cells
    mat = summed[keep]
    return (pd.DataFrame(mat.toarray() if sparse.issparse(mat) else mat), meta[keep].reset_index(drop=True))


def pseudobulk_test(X, genes, groups: pd.Series, units: pd.Series, group_a: str, group_b: str,
                    min_cells: int = 10, paired: bool = True) -> pd.DataFrame:
    """Unit-level test of group_a vs group_b on log-CPM pseudobulk profiles.

    If every unit contributes both groups and ``paired`` is True, a paired t-test
    is used (each animal/donor is its own control). Otherwise Welch's t-test across
    units. With fewer than 2 units per group no p-value is produced, which is the
    honest answer for unreplicated designs.
    """
    mask = groups.astype(str).isin([group_a, group_b]).to_numpy()
    mat, meta = pseudobulk(X[mask], groups[mask].reset_index(drop=True), units[mask].reset_index(drop=True), min_cells)
    lib = mat.sum(axis=1).to_numpy()
    lib[lib == 0] = 1
    logcpm = np.log1p(mat.to_numpy() / lib[:, None] * 1e6)
    a = meta["group"] == group_a
    b = meta["group"] == group_b
    ua, ub = set(meta.loc[a, "unit"]), set(meta.loc[b, "unit"])
    shared = sorted(ua & ub)
    res = pd.DataFrame({"gene": list(genes)})
    if paired and len(shared) >= 2:
        ia = [meta.index[(meta.unit == u) & a][0] for u in shared]
        ib = [meta.index[(meta.unit == u) & b][0] for u in shared]
        diff = logcpm[ia] - logcpm[ib]
        t, p = stats.ttest_rel(logcpm[ia], logcpm[ib], axis=0)
        res["log2fc"] = diff.mean(axis=0) / np.log(2)
        res["test"] = "paired_t_pseudobulk"
        res["n_units"] = len(shared)
    elif a.sum() >= 2 and b.sum() >= 2:
        t, p = stats.ttest_ind(logcpm[a.to_numpy()], logcpm[b.to_numpy()], axis=0, equal_var=False)
        res["log2fc"] = (logcpm[a.to_numpy()].mean(0) - logcpm[b.to_numpy()].mean(0)) / np.log(2)
        res["test"] = "welch_t_pseudobulk"
        res["n_units"] = int(min(a.sum(), b.sum()))
    else:
        res["log2fc"] = np.nan
        res["stat"] = np.nan
        res["pval"] = np.nan
        res["padj"] = np.nan
        res["test"] = "insufficient_replicates"
        res["n_units"] = int(min(a.sum(), b.sum()))
        return res
    res["stat"] = t
    res["pval"] = p
    res["padj"] = bh(p)
    return res
