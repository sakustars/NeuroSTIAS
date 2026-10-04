"""Download the 12 spatialLIBD DLPFC Visium sections (Maynard et al. 2021, Nat Neurosci) with manual layer labels.

Sources: count matrices from the spatialLIBD AWS bucket; spot positions and the
barcode-level manual layer annotation from github.com/LieberInstitute/HumanPilot.
Output: data/dlpfc/<sample>.h5ad (obs: layer, donor, section; obsm: spatial in
full-resolution pixels) and data/dlpfc/dlpfc_all.h5ad. Spots without a manual
layer label are kept but labelled NA (excluded from ARI by the experiments).
"""
import argparse
import io
import sys
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

SAMPLES = {"151507": "Br5292", "151508": "Br5292", "151509": "Br5292", "151510": "Br5292",
           "151669": "Br5595", "151670": "Br5595", "151671": "Br5595", "151672": "Br5595",
           "151673": "Br8100", "151674": "Br8100", "151675": "Br8100", "151676": "Br8100"}
H5 = "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/{s}_filtered_feature_bc_matrix.h5"
POS = "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/10X/{s}/tissue_positions_list.txt"
LAYERS = "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/10X/barcode_level_layer_map.tsv"


def get(url, dest: Path):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print("download", url)
        req = urllib.request.Request(url, headers={"User-Agent": "neurostias/0.1"})
        with urllib.request.urlopen(req, timeout=300) as r:
            dest.write_bytes(r.read())
    return dest


def main(out: Path):
    raw = out / "raw"
    lay = pd.read_csv(get(LAYERS, raw / "barcode_level_layer_map.tsv"), sep="\t", header=None,
                      names=["barcode", "sample", "layer"], dtype=str)
    adatas = []
    for s, donor in SAMPLES.items():
        a = sc.read_10x_h5(get(H5.format(s=s), raw / f"{s}_filtered_feature_bc_matrix.h5"))
        a.var_names_make_unique()
        pos = pd.read_csv(get(POS.format(s=s), raw / f"{s}_tissue_positions_list.txt"), header=None,
                          names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]).set_index("barcode")
        pos = pos.loc[a.obs_names]
        a.obsm["spatial"] = pos[["pxl_col", "pxl_row"]].to_numpy(float)
        a.obs[["array_row", "array_col"]] = pos[["array_row", "array_col"]].to_numpy()
        l = lay[lay["sample"] == s].set_index("barcode")["layer"]
        a.obs["layer"] = pd.Categorical(l.reindex(a.obs_names).fillna("NA").to_numpy())
        a.obs["section"] = s
        a.obs["donor"] = donor
        a.obs["replicate"] = donor
        a.layers["counts"] = a.X.copy()
        a.uns["source"] = {"h5": H5.format(s=s), "positions": POS.format(s=s), "layers": LAYERS,
                           "citation": "Maynard et al. 2021 Nat Neurosci 24:425-436"}
        a.write_h5ad(out / f"{s}.h5ad")
        print(s, a.shape, a.obs["layer"].value_counts().to_dict())
        a.obs_names = [f"{s}_{b}" for b in a.obs_names]
        adatas.append(a)
    allx = ad.concat(adatas, join="inner", merge="same")
    allx.write_h5ad(out / "dlpfc_all.h5ad")
    print("combined", allx.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    main(Path(p.parse_args().out))
