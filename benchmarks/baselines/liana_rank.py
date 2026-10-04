"""LIANA consensus ranking (Dimitrov et al. 2022) in the baseline environment (non-spatial, cluster level).

Usage: python liana_rank.py --input x.h5ad --groupby label --out res.csv
Input X: log-normalised expression; uses LIANA's default 'consensus' resource (mouse via orthology when --mouse).
"""
import argparse

import scanpy as sc

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--groupby", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mouse", action="store_true")
    a = p.parse_args()
    import liana as li
    ad = sc.read_h5ad(a.input)
    kw = {"resource_name": "mouseconsensus"} if a.mouse else {}
    li.mt.rank_aggregate(ad, groupby=a.groupby, use_raw=False, verbose=False, n_perms=200, **kw)
    ad.uns["liana_res"].to_csv(a.out, index=False)
    print(ad.uns["liana_res"].shape)
