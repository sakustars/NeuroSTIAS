"""PASTE pairwise alignment (Zeira et al. 2022) in the baseline environment.

Usage: python paste_align.py --target t.h5ad --source s.h5ad --alpha 0.1 --out aligned_source_xy.npy
Both inputs: counts in X, positions in obsm['spatial']. Output: source positions mapped into
the target frame (PASTE generalized Procrustes from the OT plan).
"""
import argparse
import time

import numpy as np
import scanpy as sc

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import paste as pst
    t = sc.read_h5ad(a.target)
    s = sc.read_h5ad(a.source)
    common = t.var_names.intersection(s.var_names)
    t, s = t[:, common].copy(), s[:, common].copy()
    sc.pp.filter_genes(t, min_counts=1)
    s = s[:, t.var_names].copy()
    t0 = time.time()
    pi = pst.pairwise_align(t, s, alpha=a.alpha, verbose=False)
    new = pst.stack_slices_pairwise([t, s], [pi])
    np.save(a.out, np.asarray(new[1].obsm["spatial"]))
    np.save(a.out.replace(".npy", "_target.npy"), np.asarray(new[0].obsm["spatial"]))
    print(f"elapsed {time.time() - t0:.1f}")
