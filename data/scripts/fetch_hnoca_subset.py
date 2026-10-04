"""Extract an organoid time-course subset from the Human Neural Organoid Cell Atlas (He et al. 2024 Nature)
without downloading the 19 GB file: HDF5 is read over HTTP range requests.

Selection (documented in the output's uns['selection']): for every publication with
>= 4 distinct organoid ages, report ages and sample counts; choose the
publication with the most distinct ages (ties: most samples); from each of its
bio_samples take up to ``--per-sample`` consecutive cells. Raw counts (raw/X)
are extracted. Cell annotations (annot_level_1-4, region) are HNOCA's harmonised
labels and serve as ground truth.
"""
import argparse
import json
import time
from pathlib import Path

import aiohttp
import anndata as ad
import fsspec
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

URL = "https://datasets.cellxgene.cziscience.com/127a3c9a-7bcb-4362-a73e-e60fd67f5bae.h5ad"
COLS = ["organoid_age_days", "publication", "bio_sample", "annot_level_1", "annot_level_2", "annot_level_3_rev2",
        "annot_region_rev2", "cell_line", "assay", "state_exact", "batch"]


# aiohttp's default 300 s *total* timeout aborts large range reads on slow links; keep only a stall timeout
BLOCK = 2**18  # 256 KB HTTP blocks: data/indices chunks are ~8k elements (tens of KB compressed)
HTTP_KW = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=None, sock_connect=120, sock_read=300)}}


def open_h5(block_size):
    return h5py.File(fsspec.open(URL, block_size=block_size, cache_type="blockcache", **HTTP_KW).open(), "r")


def with_retries(fn, what, attempts=8):
    for k in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network errors surface as many exception types
            print(f"retry {what} ({k + 1}/{attempts}): {type(exc).__name__}", flush=True)
            time.sleep(min(60, 10 * (k + 1)))
    raise RuntimeError(f"failed after {attempts} attempts: {what}")


def read_col(g):
    if isinstance(g, h5py.Dataset):
        return g[:]
    cats = g["categories"][:]
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats])
    codes = g["codes"][:]
    return np.where(codes >= 0, cats[np.clip(codes, 0, None)], None)


def runs(idx):
    idx = np.sort(idx)
    br = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]] + 1
    return list(zip(starts, ends))


def main(out: Path, per_sample: int, publication: str | None):
    # metadata: small blocks (HDF5 metadata/chunks are scattered); counts: large blocks (contiguous runs)
    h = open_h5(2**20)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "obs_cache.pkl"
    if cache.exists():
        obs = pd.read_pickle(cache)
        print("obs loaded from cache", obs.shape, flush=True)
    else:
        cols = {}
        for c in COLS:
            cols[c] = read_col(h["obs"][c])
            print("read column", c, flush=True)
        obs = pd.DataFrame(cols)
        obs.to_pickle(cache)
    print("obs read", obs.shape, flush=True)
    stats = (obs.groupby("publication").agg(n_ages=("organoid_age_days", "nunique"),
                                             n_samples=("bio_sample", "nunique"), n_cells=("bio_sample", "size"))
             .sort_values(["n_ages", "n_samples"], ascending=False))
    print(stats.head(15), flush=True)
    pub = publication or stats.index[0]
    sel = []
    for s, g in obs[obs.publication == pub].groupby("bio_sample"):
        r = runs(g.index.to_numpy())
        first = np.arange(r[0][0], min(r[0][1], r[0][0] + per_sample))
        sel.append(first)
    sel = np.sort(np.concatenate(sel))
    print("publication", pub, "cells selected", len(sel), "runs", len(runs(sel)), flush=True)
    # counts: each contiguous run is fetched with retries on a fresh handle and cached on disk,
    # so an interrupted extraction resumes instead of starting over
    run_dir = out / "runs_cache"
    run_dir.mkdir(exist_ok=True)
    # raw/X is gzip-chunked in small chunks (indptr: 1,251 rows/chunk, scattered over ~9 GB of the file), so large
    # HTTP blocks over-fetch enormously; read with small blocks and only the indptr entries of the selected runs
    state = {"h": open_h5(BLOCK)}

    def reopen():
        state["h"] = open_h5(BLOCK)

    run_list = runs(sel)
    ip = run_dir / "indptr_selected_runs.npz"
    if not ip.exists():
        parts = {f"{a}_{b}": with_retries(lambda a=a, b=b: state["h"]["raw"]["X"]["indptr"][a:b + 1], f"indptr {a}")
                 for a, b in run_list}
        np.savez(ip, **parts)
    ipz = np.load(ip)
    n_genes = int(with_retries(lambda: state["h"]["raw"]["X"].attrs["shape"][1], "shape"))
    rows, data_parts, ind_parts, ptr = [], [], [], [0]
    for ri, (a, b) in enumerate(run_list):
        cf = run_dir / f"run_{a}_{b}.npz"
        run_ptr = ipz[f"{a}_{b}"]
        lo, hi = run_ptr[0], run_ptr[-1]
        if not cf.exists():
            print("reading run", ri, "of", len(run_list), flush=True)

            def get(lo=lo, hi=hi):
                try:
                    Xg = state["h"]["raw"]["X"]
                    return Xg["data"][lo:hi], Xg["indices"][lo:hi]
                except Exception:
                    reopen()
                    raise
            d, i = with_retries(get, f"run {ri}")
            np.savez(cf, data=d, indices=i)
        z = np.load(cf)
        local = run_ptr - lo
        data_parts.append(z["data"])
        ind_parts.append(z["indices"])
        for k in range(b - a):
            ptr.append(ptr[-1] + (local[k + 1] - local[k]))
    X = sparse.csr_matrix((np.concatenate(data_parts), np.concatenate(ind_parts), np.array(ptr)),
                          shape=(len(sel), n_genes))
    hv = state["h"]["raw"]["var"]
    var = pd.DataFrame({"gene_id": with_retries(lambda: read_col(hv["_index"]), "var index") if "_index" in hv else None})
    if "feature_name" in hv:
        var["gene_symbol"] = with_retries(lambda: read_col(hv["feature_name"]), "var symbols")
    a = ad.AnnData(X=X.astype(np.float32), obs=obs.iloc[sel].reset_index(drop=True), var=var)
    a.obs_names = [f"hnoca_{i}" for i in sel]
    a.var_names = a.var["gene_symbol"].astype(str) if "gene_symbol" in a.var else a.var["gene_id"].astype(str)
    a.var.index.name = None  # avoid clash with the 'gene_symbol' column when writing
    a.var_names_make_unique()
    a.layers["counts"] = a.X.copy()
    a.uns["selection"] = json.dumps({"source": URL, "publication": pub, "per_sample": per_sample,
                                     "rows": "first contiguous run per bio_sample", "n_cells": int(len(sel)),
                                     "citation": "He et al. 2024 Nature 635 (HNOCA)"})
    out.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out / "hnoca_publication_stats.csv")
    a.write_h5ad(out / "hnoca_timecourse_subset.h5ad")
    print("saved", a.shape, a.obs.organoid_age_days.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--per-sample", type=int, default=600)
    p.add_argument("--publication", default=None)
    a = p.parse_args()
    main(Path(a.out), a.per_sample, a.publication)
