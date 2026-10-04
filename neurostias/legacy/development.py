"""Stemness (V4.4.1) and neurodevelopmental stage scoring.

``stemness_v441`` reproduces V4.4.1: min–max-scaled mean expression of
pluripotency / ESC / self-renewal / differentiation gene sets, composite with
weights 2 (pluripotency, self-renewal, core) and 1 (inverted differentiation).
It is a heuristic gene-set score, not CytoTRACE or OCLR-mRNAsi. Pluripotency
genes are rarely informative in brain tissue, so for neuroscience the default is
``neurodevelopment_stages``: per-observation enrichment against
expression-matched control gene sets (``atlas.annotate.score_panels``) for
radial glia/NPC, intermediate progenitor, cycling, immature neuron, mature
neuron and glial progenitor programmes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ..atlas import markers
from ..atlas.annotate import call_labels, score_panels
from ..core.io import detect_species, gene_symbols

STEMNESS_SETS_V441 = {
    "pluripotency_core": ["POU5F1", "NANOG", "SOX2", "KLF4", "MYC", "LIN28A", "LIN28B", "SALL4", "ESRRB", "UTF1"],
    "esc_signature": ["DNMT3B", "GDF3", "GABRB3", "ZFP42", "TDGF1", "PODXL", "THY1", "CD9", "FUT4", "PROM1"],
    "self_renewal": ["TERT", "BMI1", "EZH2", "SUZ12", "EED", "WNT3", "WNT3A", "FGF2", "FGF4", "TGFB1"],
    "differentiation": ["GATA6", "SOX17", "FOXA2", "CDX2", "MIXL1", "EOMES", "TBXT", "PAX6", "OTX2", "HOXB1"],
}


def stemness_v441(adata, min_genes_per_set: int = 3) -> dict:
    syms = gene_symbols(adata).astype(str).str.upper()
    lookup = {g: i for i, g in enumerate(syms)}
    X = adata.layers["log_normalized"] if "log_normalized" in adata.layers else adata.X
    used = {}
    for name, genes in STEMNESS_SETS_V441.items():
        idx = [lookup[g] for g in genes if g in lookup]
        if len(idx) < min_genes_per_set:
            continue
        v = X[:, idx]
        v = np.asarray(v.mean(axis=1)).ravel() if sparse.issparse(v) else np.asarray(v).mean(axis=1)
        if v.max() > v.min():
            v = (v - v.min()) / (v.max() - v.min())
        adata.obs[f"stemness_{name}"] = v
        used[name] = len(idx)
    if not used:
        return {"status": "not_applicable", "reason": "no stemness gene set has enough genes in this dataset"}
    cols = [f"stemness_{k}" for k in used]
    S = adata.obs[cols].to_numpy().copy()
    w = np.ones(len(cols))
    for i, c in enumerate(cols):
        if any(t in c for t in ("pluripotency", "self_renewal", "core")):
            w[i] = 2.0
        elif "differentiation" in c:
            S[:, i] = 1.0 - S[:, i]
    comp = np.average(S, axis=1, weights=w / w.sum())
    if comp.max() > comp.min():
        comp = (comp - comp.min()) / (comp.max() - comp.min())
    adata.obs["stemness_composite"] = comp
    return {"status": "completed", "gene_sets_used": used}


def neurodevelopment_stages(adata, species: str = "auto", alpha: float = 0.05, n_null: int = 1000,
                            seed: int = 0) -> dict:
    if species == "auto":
        species = detect_species(gene_symbols(adata))
    eff, pv, cov = score_panels(adata, markers.panels("neurodevelopment", species), n_null=n_null, seed=seed)
    for c in eff.columns:
        adata.obs[f"devstage_{c.split(' /')[0].replace(' ', '_').lower()}"] = eff[c].to_numpy()
    lab, _ = call_labels(eff, pv, alpha)
    adata.obs["neurodev_stage"] = pd.Categorical(lab)
    # maturation index: mature-neuron effect minus progenitor effect (both standardised)
    mat = None
    if {"Mature neuron", "Radial glia / NPC"} <= set(eff.columns):
        mat = eff["Mature neuron"] - eff["Radial glia / NPC"]
        adata.obs["maturation_index"] = mat.to_numpy()
    return {"status": "completed", "stage_counts": pd.Series(lab).value_counts().to_dict(),
            "coverage": cov, "has_maturation_index": mat is not None}
