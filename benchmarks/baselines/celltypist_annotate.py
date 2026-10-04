"""CellTypist (Dominguez Conde et al. 2022) annotation in the baseline environment.

Usage: python celltypist_annotate.py --input x.h5ad --model Mouse_Whole_Brain.pkl --out labels.csv [--majority]
Input X: log1p(CP10k) expression (CellTypist's expected input).
"""
import argparse

import pandas as pd
import scanpy as sc

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--majority", action="store_true")
    a = p.parse_args()
    import celltypist
    from celltypist import models
    models.download_models(model=[a.model], force_update=False)
    ad = sc.read_h5ad(a.input)
    pred = celltypist.annotate(ad, model=a.model, majority_voting=a.majority)
    lab = pred.predicted_labels
    col = "majority_voting" if a.majority and "majority_voting" in lab else "predicted_labels"
    out = pd.DataFrame({"obs": ad.obs_names, "label": lab[col].astype(str).to_numpy(),
                        "conf": pred.probability_matrix.max(axis=1).to_numpy()})
    out.to_csv(a.out, index=False)
    m = models.Model.load(a.model)
    print("genes_in_model", len(m.features), "overlap", len(set(m.features) & set(ad.var_names)))
