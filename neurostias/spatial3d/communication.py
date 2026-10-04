"""Neuro-specific, distance-constrained, replicate-aware cell–cell communication in 2D/3D.

Ligands of small-molecule transmitters are defined by their synthesis enzymes and
vesicular transporters, not by a single "ligand gene" (following NeuronChat,
Zhao et al. 2023). Gene groups use ``;`` for required components and ``|`` for
alternatives: ``GAD1|GAD2;SLC32A1`` means (GAD1 or GAD2) and SLC32A1. Abundance of
a ligand/receptor is the geometric mean over required groups of the maximum over
alternatives.

Scoring (per interaction, sender type A → receiver type B):

* edges come from a within-sample kNN graph on anisotropy-scaled 3D coordinates
  (``core.graph``), so communication is restricted to physical neighbours;
* edge score = ligand(sender) × receptor(receiver);
* statistic = mean edge score over A→B edges;
* null = receptor values permuted *within each independent sample* (so the
  null preserves each sample's expression distribution); p-values use a normal
  approximation of the permutation null (empirical p-values are also reported);
* replicate consistency = fraction of samples in which A→B exceeds its own
  within-sample null 95th percentile. A pooled p-value alone is not reported as
  significant unless it is consistent across samples (``min_replicate_fraction``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm

from ..core import graph
from ..core.io import detect_species, gene_symbols
from ..core.stats import bh

NEURO_DB = Path(__file__).resolve().parent / "neuro_lr.tsv"

# V4.4.1 generic curated pairs (kept as the "generic" baseline for comparison)
GENERIC_PAIRS = [
    ("TGFB1", "TGFBR1"), ("TGFB1", "TGFBR2"), ("TGFB2", "TGFBR2"), ("VEGFA", "FLT1"), ("VEGFA", "KDR"),
    ("VEGFB", "FLT1"), ("FGF2", "FGFR1"), ("FGF7", "FGFR2"), ("FGF10", "FGFR2"), ("PDGFA", "PDGFRA"),
    ("PDGFB", "PDGFRB"), ("EGF", "EGFR"), ("HBEGF", "EGFR"), ("AREG", "EGFR"), ("EREG", "EGFR"),
    ("WNT1", "FZD1"), ("WNT3A", "FZD7"), ("WNT5A", "FZD5"), ("JAG1", "NOTCH1"), ("JAG1", "NOTCH2"),
    ("DLL1", "NOTCH1"), ("DLL4", "NOTCH1"), ("CXCL12", "CXCR4"), ("CXCL8", "CXCR1"), ("CCL2", "CCR2"),
    ("CCL5", "CCR5"), ("CSF1", "CSF1R"), ("IL6", "IL6R"), ("IL10", "IL10RA"), ("TNF", "TNFRSF1A"),
    ("IFNG", "IFNGR1"), ("KITLG", "KIT"), ("HGF", "MET"), ("GAS6", "AXL"), ("ANGPT1", "TEK"),
    ("SEMA3A", "NRP1"), ("SPP1", "CD44"), ("ICAM1", "ITGAL"), ("VCAM1", "ITGA4"), ("MIF", "CD74"),
]


def load_database(name: str = "neuro") -> pd.DataFrame:
    if name == "neuro":
        return pd.read_csv(NEURO_DB, sep="\t").fillna("")
    if name == "generic_v441":
        return pd.DataFrame([{"interaction": f"{l}_{r}", "category": "generic", "ligand": l, "ligand_genes": l,
                              "receptor": r, "receptor_genes": r, "note": "3D-STIAS V4.4.1 curated pair"}
                             for l, r in GENERIC_PAIRS])
    raise KeyError(name)


def _to_species(sym: str, species: str) -> str:
    return sym if species == "human" else sym[:1].upper() + sym[1:].lower()


def _abundance(spec: str, X, lookup: dict[str, int], species: str) -> np.ndarray | None:
    """Geometric mean over ';' groups of max over '|' alternatives. None if a required group is absent."""
    groups = []
    for grp in spec.split(";"):
        idx = [lookup[_to_species(g, species).upper()] for g in grp.split("|") if _to_species(g, species).upper() in lookup]
        if not idx:
            return None
        sub = X[:, idx]
        sub = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
        groups.append(np.maximum(sub.max(axis=1), 0))
    G = np.vstack(groups)
    return np.exp(np.log(G + 1e-12).mean(axis=0)) * (G.min(axis=0) > 0)


def communication(adata, label_key: str, database: str = "neuro", k: int = 8, z_scale: float = 1.0,
                  block_key: str | None = None, section_key: str | None = None, n_permutations: int = 200,
                  min_edges: int = 20, min_replicate_fraction: float = 0.5, species: str = "auto",
                  seed: int = 0) -> pd.DataFrame:
    if species == "auto":
        species = detect_species(gene_symbols(adata))
    db = load_database(database)
    X = adata.layers["log_normalized"] if "log_normalized" in adata.layers else adata.X
    syms = gene_symbols(adata).astype(str).str.upper()
    lookup = {}
    for i, g in enumerate(syms):
        lookup.setdefault(g, i)
    coords = np.asarray(adata.obsm["spatial_3d"] if "spatial_3d" in adata.obsm else adata.obsm["spatial"], dtype=float)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])
    blocks, bkey = graph.block_labels(adata.obs, block_key)
    sections = adata.obs[section_key].astype(str).to_numpy() if section_key else None
    W = graph.knn_graph(coords, blocks, k=k, z_scale=z_scale, sections=sections).tocoo()
    src, dst = W.row, W.col
    labels = adata.obs[label_key].astype(str).to_numpy()
    pair = pd.Series(labels[src]) + " -> " + pd.Series(labels[dst])
    codes, groups = pd.factorize(pair)
    n_groups = len(groups)
    counts = np.bincount(codes, minlength=n_groups)
    edge_block = blocks[src]
    ublocks = np.unique(blocks)
    block_rows = [np.flatnonzero(blocks == b) for b in ublocks]
    rng = np.random.default_rng(seed)
    rows = []
    for rec in db.itertuples(index=False):
        L = _abundance(rec.ligand_genes, X, lookup, species)
        R = _abundance(rec.receptor_genes, X, lookup, species)
        if L is None or R is None or L.max() == 0 or R.max() == 0:
            continue
        e = L[src] * R[dst]
        obs = np.bincount(codes, weights=e, minlength=n_groups) / np.maximum(counts, 1)
        exceed = np.zeros(n_groups)
        s1 = np.zeros(n_groups)
        s2 = np.zeros(n_groups)
        rep_null = {b: [] for b in ublocks}
        for _ in range(n_permutations):
            Rp = R.copy()
            for bi in block_rows:
                Rp[bi] = R[rng.permutation(bi)]
            ep = L[src] * Rp[dst]
            null = np.bincount(codes, weights=ep, minlength=n_groups) / np.maximum(counts, 1)
            exceed += null >= obs
            s1 += null
            s2 += null ** 2
            for b in ublocks:
                m = edge_block == b
                cb = np.bincount(codes[m], minlength=n_groups)
                rep_null[b].append(np.bincount(codes[m], weights=ep[m], minlength=n_groups) / np.maximum(cb, 1))
        p_emp = (1 + exceed) / (n_permutations + 1)
        mu = s1 / n_permutations
        sd = np.sqrt(np.maximum(s2 / n_permutations - mu ** 2, 0))
        # Normal approximation of the permutation null (means over >= min_edges edges): avoids the
        # 1/(n_permutations+1) resolution floor, which otherwise makes BH significance unreachable
        # when thousands of interaction x cell-type tests are corrected together.
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(sd > 0, norm.sf((obs - mu) / sd), np.where(obs > mu, 0.0, 1.0))
        rep_hits = np.zeros(n_groups)
        rep_tested = np.zeros(n_groups)
        for b in ublocks:
            m = edge_block == b
            cb = np.bincount(codes[m], minlength=n_groups)
            ob = np.bincount(codes[m], weights=e[m], minlength=n_groups) / np.maximum(cb, 1)
            q95 = np.quantile(np.vstack(rep_null[b]), 0.95, axis=0)
            tested = cb >= min_edges
            rep_tested += tested
            rep_hits += tested & (ob > q95)
        for gi, g in enumerate(groups):
            if counts[gi] < min_edges or obs[gi] <= 0:
                continue
            sender, receiver = g.split(" -> ")
            rows.append({"interaction": rec.interaction, "category": rec.category, "ligand": rec.ligand,
                         "receptor": rec.receptor, "sender": sender, "receiver": receiver,
                         "mean_score": float(obs[gi]), "n_edges": int(counts[gi]), "pval": float(p[gi]),
                         "pval_empirical": float(p_emp[gi]),
                         "n_replicates_tested": int(rep_tested[gi]), "n_replicates_supporting": int(rep_hits[gi])})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["padj"] = bh(res["pval"])
    res["replicate_fraction"] = res["n_replicates_supporting"] / res["n_replicates_tested"].clip(lower=1)
    multi = len(ublocks) > 1
    res["significant"] = (res["padj"] < 0.05) & ((res["replicate_fraction"] >= min_replicate_fraction) if multi else True)
    res.attrs.update({"database": database, "block_key": bkey, "n_blocks": int(len(ublocks)), "z_scale": z_scale})
    return res.sort_values(["significant", "padj", "mean_score"], ascending=[False, True, False]).reset_index(drop=True)
