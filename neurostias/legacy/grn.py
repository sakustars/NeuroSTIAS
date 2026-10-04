"""Exploratory TF–target co-expression network (ported from V4.4.1 ``gene_regulatory_network_3d``).

Method (unchanged from V4.4.1 except for the TF list): known TFs present in the
data are ranked by variance; targets are the most variable non-TF genes.
Spearman correlations are computed globally or stratified by cell type, BH-FDR
is applied once over the complete family of tests, and edges require
|rho| >= ``corr_threshold`` and FDR < ``fdr_threshold``. Each edge is annotated with
a spatial-consistency correlation of neighbourhood-smoothed expression.

This is co-expression, not causal regulation (it is not SCENIC). The default TF
list is neural (``tf_set='neural'``); ``tf_set='generic_v441'`` restores the
original list.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse, stats

from ..core import graph
from ..core.stats import bh

GENERIC_V441_TFS = {
    'AR', 'ARNT', 'ATF1', 'ATF2', 'ATF3', 'ATF4', 'BATF', 'BCL6', 'CEBPA', 'CEBPB', 'CREB1', 'CTCF', 'E2F1', 'E2F2',
    'E2F3', 'ELF1', 'ELK1', 'ESR1', 'ETS1', 'ETV4', 'FOS', 'FOSB', 'FOXA1', 'FOXA2', 'FOXC1', 'FOXP1', 'GATA1',
    'GATA2', 'GATA3', 'HIF1A', 'IRF1', 'IRF3', 'IRF4', 'IRF7', 'JUN', 'JUNB', 'JUND', 'KLF2', 'KLF4', 'KLF5', 'KLF6',
    'MAF', 'MAFB', 'MAX', 'MEF2A', 'MEF2C', 'MITF', 'MYC', 'MYCN', 'NANOG', 'NFE2L2', 'NFATC1', 'NFATC2', 'NFKB1',
    'NFKB2', 'NOTCH1', 'NR3C1', 'PAX5', 'POU2F1', 'POU2F2', 'POU5F1', 'PPARG', 'RARA', 'RELA', 'RUNX1', 'RUNX2',
    'RUNX3', 'SMAD2', 'SMAD3', 'SMAD4', 'SOX2', 'SOX4', 'SOX9', 'SP1', 'SPI1', 'STAT1', 'STAT2', 'STAT3', 'STAT4',
    'STAT5A', 'STAT5B', 'STAT6', 'TBX21', 'TCF3', 'TCF4', 'TCF7', 'TEAD1', 'TEAD4', 'TP53', 'TWIST1', 'XBP1', 'YY1',
    'ZEB1'}

# Neural transcription factors: cortical identity, interneuron specification,
# glial lineage, activity-dependent and neurodevelopmental regulators.
NEURAL_TFS = {
    'NEUROD1', 'NEUROD2', 'NEUROD6', 'NEUROG2', 'TBR1', 'EOMES', 'SATB2', 'BCL11B', 'FEZF2', 'CUX1', 'CUX2', 'RORB',
    'POU3F2', 'POU3F3', 'TLE4', 'FOXP2', 'SOX5', 'EMX1', 'EMX2', 'FOXG1', 'PAX6', 'SOX2', 'SOX11', 'SOX4', 'LHX2',
    'DLX1', 'DLX2', 'DLX5', 'DLX6', 'LHX6', 'LHX8', 'NKX2-1', 'SOX6', 'MAF', 'MAFB', 'PROX1', 'NR2F2', 'SP8', 'GSX2',
    'ASCL1', 'OLIG1', 'OLIG2', 'SOX10', 'MYRF', 'NKX2-2', 'NKX6-2', 'ZNF488', 'SOX8', 'SOX9', 'NFIA', 'NFIB', 'NFIX',
    'GLI3', 'HES1', 'HES5', 'ID3', 'SPI1', 'IRF8', 'RUNX1', 'MEF2C', 'SALL1', 'FOS', 'FOSB', 'JUNB', 'EGR1', 'EGR3',
    'NPAS4', 'NR4A1', 'NR4A2', 'ARX', 'ISL1', 'LMO4', 'ZEB2', 'TCF4', 'TCF7L2', 'GATA3', 'OTX2', 'EN1', 'EN2',
    'LMX1A', 'LMX1B', 'FOXA2', 'PITX3', 'PHOX2B', 'HOXB5', 'ESRRG', 'RFX4', 'ZIC1', 'ZIC2'}


def _sym(adata):
    from ..core.io import gene_symbols
    return gene_symbols(adata).astype(str)


def tf_coexpression_network(adata, n_tfs: int = 20, n_targets: int = 50, corr_threshold: float = 0.2,
                            fdr_threshold: float = 0.05, context_key: str | None = None, min_cells_per_context: int = 50,
                            tf_set: str = "neural", block_key: str | None = None) -> dict:
    tfs_known = NEURAL_TFS if tf_set == "neural" else GENERIC_V441_TFS
    names = _sym(adata)
    X = adata.layers["log_normalized"] if "log_normalized" in adata.layers else adata.X
    if sparse.issparse(X):
        mean = np.asarray(X.mean(0)).ravel()
        var = np.asarray(X.power(2).mean(0)).ravel() - mean ** 2
    else:
        var = np.var(np.asarray(X), axis=0)
    upper = names.str.upper()
    tf_idx = [i for i, g in enumerate(upper) if g in tfs_known and var[i] > 0]
    tf_idx = sorted(tf_idx, key=lambda i: var[i], reverse=True)[:n_tfs]
    if len(tf_idx) < 2:
        return {"status": "not_applicable", "reason": f"fewer than 2 known {tf_set} TFs expressed; no substitute genes used",
                "edges": pd.DataFrame(), "tests": pd.DataFrame()}
    tf_set_idx = set(tf_idx)
    tgt_idx = [i for i in np.argsort(var)[::-1] if i not in tf_set_idx and var[i] > 0][:n_targets]
    cols = tf_idx + tgt_idx
    E = X[:, cols]
    E = E.toarray() if sparse.issparse(E) else np.asarray(E, dtype=float)
    tf_n, tg_n = list(names[tf_idx]), list(names[tgt_idx])
    contexts = [("global", np.ones(adata.n_obs, bool))]
    if context_key:
        lab = adata.obs[context_key].astype(str).to_numpy()
        vc = pd.Series(lab).value_counts()
        ctx = [c for c in vc.index if vc[c] >= min_cells_per_context]
        if len(ctx) >= 2:
            contexts = [(c, lab == c) for c in sorted(ctx)]
    rows = []
    nt = len(tf_idx)
    for cname, m in contexts:
        rho, p = stats.spearmanr(E[m], axis=0)
        rho = np.atleast_2d(rho)
        p = np.atleast_2d(p)
        for i in range(nt):
            for j in range(len(tgt_idx)):
                r_, p_ = rho[i, nt + j], p[i, nt + j]
                rows.append({"context": cname, "n_cells": int(m.sum()), "source": tf_n[i], "target": tg_n[j],
                             "rho": float(np.nan_to_num(r_)), "pval": float(np.nan_to_num(p_, nan=1.0))})
    tests = pd.DataFrame(rows)
    tests["fdr"] = bh(tests["pval"])
    tests["significant"] = (tests["rho"].abs() >= corr_threshold) & (tests["fdr"] < fdr_threshold)
    edges = (tests[tests["significant"]].assign(abs_rho=lambda d: d["rho"].abs())
             .sort_values(["source", "target", "fdr", "abs_rho"], ascending=[True, True, True, False])
             .drop_duplicates(["source", "target"]).drop(columns="abs_rho").reset_index(drop=True))
    if len(edges):
        coords = np.asarray(adata.obsm["spatial_3d"] if "spatial_3d" in adata.obsm else adata.obsm["spatial"], dtype=float)
        if coords.shape[1] == 2:
            coords = np.column_stack([coords, np.zeros(len(coords))])
        blocks, _ = graph.block_labels(adata.obs, block_key)
        Wn = graph.row_normalize(graph.knn_graph(coords, blocks, k=6))
        S = Wn @ E
        pos = {g: k for k, g in enumerate(tf_n + tg_n)}
        sp = [stats.spearmanr(S[:, pos[s]], S[:, pos[t]])[0] for s, t in zip(edges["source"], edges["target"])]
        edges["spatial_rho"] = np.nan_to_num(sp)
    G = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.DiGraph) if len(edges) else nx.DiGraph()
    return {"status": "completed", "tf_set": tf_set, "n_tfs": nt, "n_targets": len(tgt_idx),
            "n_edges": int(G.number_of_edges()), "n_contexts": len(contexts),
            "edges": edges, "tests": tests,
            "density": float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0}
