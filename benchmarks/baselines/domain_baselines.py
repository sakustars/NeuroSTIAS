"""Spatial-domain baselines run in the isolated baseline environment.

Usage: python domain_baselines.py --input section.h5ad --method {banksy,spagcn,graphst} --k 7 --seed 0 --out labels.csv
Input must contain raw counts in X (or layers['counts']) and obsm['spatial'].
Each method follows its published tutorial defaults; the cluster number is matched
to k by resolution search where the method uses Leiden/Louvain.
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

# SpaGCN 1.2.7 uses the `.A` dense alias that SciPy >= 1.14 removed; restore it (behaviour-identical).
for _cls in (sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix):
    if not hasattr(_cls, "A"):
        _cls.A = property(lambda self: self.toarray())


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
    except Exception:
        pass


def leiden_k(adata, k, use_rep=None, key="clust", seed=0):
    lo, hi = 0.01, 3.0
    best = None
    for _ in range(30):
        res = (lo + hi) / 2
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
        n = adata.obs[key].nunique()
        best = adata.obs[key].astype(str).to_numpy()
        if n == k:
            break
        if n < k:
            lo = res
        else:
            hi = res
    return best


def prep(adata):
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    adata.layers["counts"] = adata.X.copy()
    return adata


def run_banksy(adata, k, seed):
    from banksy.initialize_banksy import initialize_banksy
    from banksy.embed_banksy import generate_banksy_matrix
    from banksy_utils.umap_pca import pca_umap
    from banksy.cluster_methods import run_Leiden_partition
    adata = prep(adata)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    adata = adata[:, adata.var.highly_variable].copy()
    adata.obs["xc"], adata.obs["yc"] = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    coord_keys = ("xc", "yc", "spatial")
    bd = initialize_banksy(adata, coord_keys, num_neighbours=15, nbr_weight_decay="scaled_gaussian", max_m=1,
                           plt_edge_hist=False, plt_nbr_weights=False, plt_agf_angles=False, plt_theta=False)
    results_df, _ = generate_banksy_matrix(adata, bd, lambda_list=[0.8], max_m=1, verbose=False)
    pca_umap(results_df, pca_dims=[20], add_umap=False, plt_remaining_var=False)
    # resolution search to obtain k clusters
    lo, hi, labels = 0.05, 3.0, None
    for _ in range(25):
        res = (lo + hi) / 2
        rdf, _ = run_Leiden_partition(results_df, [res], num_nn=50, num_iterations=-1, partition_seed=seed,
                                      match_labels=False, verbose=False)
        lab = rdf.iloc[0]["labels"]
        dense = np.asarray(lab.dense)
        n = len(np.unique(dense))
        labels = dense
        if n == k:
            break
        lo, hi = (res, hi) if n < k else (lo, res)
    return labels, adata.obs_names


def run_spagcn(adata, k, seed):
    import SpaGCN as spg
    adata = prep(adata)
    x, y = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    adj = spg.calculate_adj_matrix(x=x, y=y, histology=False)
    spg.prefilter_genes(adata, min_cells=3)
    spg.prefilter_specialgenes(adata)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    l = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    # k is known in this benchmark, so use SpaGCN's documented k-means initialisation with n_clusters=k
    # (the louvain initialisation needs the 'louvain' package, which does not build on this platform)
    clf = spg.SpaGCN()
    clf.set_l(l)
    seed_all(seed)
    clf.train(adata, adj, init_spa=True, init="kmeans", n_clusters=k, tol=5e-3, lr=0.05, max_epochs=200)
    pred, _ = clf.predict()
    adj2d = spg.calculate_adj_matrix(x=x, y=y, histology=False)
    refined = spg.refine(sample_id=adata.obs.index.tolist(), pred=pred.tolist(), dis=adj2d, shape="hexagon")
    return np.asarray(refined), adata.obs_names


def run_graphst(adata, k, seed):
    import torch
    from GraphST import GraphST
    adata = prep(adata)
    model = GraphST.GraphST(adata, device=torch.device("cpu"), random_seed=seed)
    adata = model.train()
    from sklearn.decomposition import PCA
    emb = PCA(n_components=20, random_state=seed).fit_transform(adata.obsm["emb"])
    adata.obsm["emb_pca"] = emb
    sc.pp.neighbors(adata, use_rep="emb_pca", random_state=seed)
    return leiden_k(adata, k, seed=seed), adata.obs_names


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    seed_all(a.seed)
    ad = sc.read_h5ad(a.input)
    t0 = time.time()
    labels, names = {"banksy": run_banksy, "spagcn": run_spagcn, "graphst": run_graphst}[a.method](ad, a.k, a.seed)
    pd.DataFrame({"obs": np.asarray(names), "label": np.asarray(labels).astype(str)}).to_csv(a.out, index=False)
    print(f"elapsed {time.time() - t0:.1f}")
