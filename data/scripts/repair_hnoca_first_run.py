"""Complete the HNOCA subset from a first extraction whose write stopped after X and obs.

The first run of ``fetch_hnoca_subset.py`` wrote ``X`` and ``obs`` and then failed while
writing ``var`` (index-name clash, since fixed). Selection is deterministic (same
publication, same first ``per_sample`` cells per bio_sample), so X/obs are reused
instead of re-downloading ~590 MB, after verification against the remote atlas:

1. obs index -> original atlas row ids; selection recomputed from the cached atlas obs
   must give the same rows;
2. per-row non-zero counts of every cell must equal the remote ``raw/X/indptr`` diffs;
3. ``data`` and ``indices`` of ``--check-runs`` randomly chosen contiguous runs must be
   byte-identical to the remote values.

Only ``var`` (gene ids/symbols) is fetched. Any failed check aborts without writing.
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

import fetch_hnoca_subset as F


def main(out: Path, per_sample: int, check_runs: int, seed: int):
    src = out / "hnoca_timecourse_subset.CORRUPT_first_run.h5ad"
    with h5py.File(src, "r") as h:
        X = read_elem(h["X"])
        obs = read_elem(h["obs"])
    sel = np.array([int(s.split("_")[1]) for s in obs.index])
    # 1. selection reproduces
    full = pd.read_pickle(out / "obs_cache.pkl")
    stats = (full.groupby("publication").agg(n_ages=("organoid_age_days", "nunique"),
                                              n_samples=("bio_sample", "nunique"))
             .sort_values(["n_ages", "n_samples"], ascending=False))
    pub = stats.index[0]
    expect = np.sort(np.concatenate([np.arange(r[0][0], min(r[0][1], r[0][0] + per_sample))
                                     for _, g in full[full.publication == pub].groupby("bio_sample")
                                     for r in [F.runs(g.index.to_numpy())]]))
    assert np.array_equal(sel, expect), "selection differs from deterministic re-selection"
    assert (obs["publication"].astype(str) == pub).all()
    # 2. per-row nnz vs remote indptr
    # raw/X/indptr is stored in small gzip chunks scattered across the remote file, so only the entries of
    # the selected runs are read, with small HTTP blocks
    h = F.open_h5(F.BLOCK)
    run_list = F.runs(sel)
    run_ptr = {}
    for a, b in run_list:
        run_ptr[(a, b)] = F.with_retries(lambda a=a, b=b: h["raw"]["X"]["indptr"][a:b + 1], f"indptr {a}")
    remote_nnz = np.concatenate([np.diff(run_ptr[r]) for r in run_list])
    assert np.array_equal(np.diff(X.indptr), remote_nnz), "per-row nnz mismatch"
    print("per-row nnz identical for all", len(sel), "cells", flush=True)
    # 3. exact values on random runs
    rng = np.random.default_rng(seed)
    pos = {c: i for i, c in enumerate(sel)}
    checked = []
    for ri in rng.choice(len(run_list), size=min(check_runs, len(run_list)), replace=False):
        ra, rb = run_list[ri]
        a, b = int(ra), int(min(rb, ra + 50))  # first <=50 cells of the run
        lo, hi = run_ptr[(ra, rb)][0], run_ptr[(ra, rb)][b - a]
        d, i = F.with_retries(lambda: (h["raw"]["X"]["data"][lo:hi], h["raw"]["X"]["indices"][lo:hi]), f"run {ri}")
        r0, r1 = pos[a], pos[b - 1] + 1
        ld, li = X.data[X.indptr[r0]:X.indptr[r1]], X.indices[X.indptr[r0]:X.indptr[r1]]
        assert np.array_equal(ld, d.astype(ld.dtype)) and np.array_equal(li, i), f"values differ in run {ri}"
        checked.append([int(a), int(b)])
        print("verified run", int(ri), "cells", b - a, flush=True)
    # var
    hv = h["raw"]["var"]
    var = pd.DataFrame({"gene_id": F.with_retries(lambda: F.read_col(hv["_index"]), "var index")})
    var["gene_symbol"] = F.with_retries(lambda: F.read_col(hv["feature_name"]), "var symbols")
    assert len(var) == X.shape[1]
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.var_names = a.var["gene_symbol"].astype(str)
    a.var.index.name = None
    a.var_names_make_unique()
    a.layers["counts"] = a.X.copy()
    a.uns["selection"] = json.dumps({
        "source": F.URL, "publication": pub, "per_sample": per_sample,
        "rows": "first contiguous run per bio_sample", "n_cells": int(len(sel)),
        "citation": "He et al. 2024 Nature 635 (HNOCA)",
        "assembly": "X/obs from first extraction (write stopped at var); verified: selection identical, per-row nnz "
                    "identical for all cells, exact data/indices for randomly sampled runs; var fetched separately",
        "verified_runs": checked})
    a.write_h5ad(out / "hnoca_timecourse_subset.h5ad")
    print("saved", a.shape, a.obs.organoid_age_days.value_counts().sort_index().to_dict(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--per-sample", type=int, default=600)
    p.add_argument("--check-runs", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    x = p.parse_args()
    main(Path(x.out), x.per_sample, x.check_runs, x.seed)
