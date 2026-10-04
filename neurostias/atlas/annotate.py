"""Hierarchical brain cell-type annotation from curated marker panels.

Level 1 assigns a major class (neurons, glia, vascular, immune, ...). Level 2
assigns neuron subclasses (layer-specific excitatory types, interneuron
families) only to observations whose level-1 call is a neuron. Neurotransmitter
identity and injury states are scored separately and never overwrite the class.

Scoring is a per-observation enrichment test, not an argmax of z-scores:

* observed score = mean log-normalised expression of the panel genes;
* null = the same statistic for ``n_null`` random gene sets in which every panel
  gene is replaced by a gene from the same expression-mean bin (Seurat-style
  control genes, Tirosh et al. 2016);
* effect = (observed − null mean) / null SD, and an empirical one-sided p-value.

An observation is labelled with the panel of largest effect among panels with
p < ``alpha`` / (number of usable panels) (Bonferroni within each observation); if no panel is significant it is "Unresolved". Panels with fewer
than ``min_genes`` detected genes are reported as unusable instead of being
scored. This matters for targeted or HVG-subset data: when the neuronal panels
are absent from the gene list, the method says so rather than spreading
observations over the remaining classes by chance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ..core.io import detect_species, gene_symbols
from . import markers


def _expr(adata, layer: str | None = None):
    if layer and layer in adata.layers:
        X = adata.layers[layer]
    elif "log_normalized" in adata.layers:
        X = adata.layers["log_normalized"]
    else:
        X = adata.X
    return X if sparse.issparse(X) else sparse.csr_matrix(np.asarray(X, dtype=float))


def score_panels(adata, panel_dict: dict[str, list[str]], min_genes: int = 3, layer: str | None = None,
                 n_null: int = 1000, n_bins: int = 25, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (effect obs×panels, pvalue obs×panels, coverage table)."""
    rng = np.random.default_rng(seed)
    syms = pd.Index(gene_symbols(adata).astype(str).str.upper())
    lookup: dict[str, int] = {}
    for i, g in enumerate(syms):
        lookup.setdefault(g, i)
    X = _expr(adata, layer).tocsr()
    n_genes = X.shape[1]
    gmean = np.asarray(X.mean(axis=0)).ravel()
    ranks = pd.Series(gmean).rank(method="first").to_numpy()
    bins = np.minimum((ranks - 1) * n_bins // n_genes, n_bins - 1).astype(int)
    all_panel_idx = np.array(sorted({lookup[g.upper()] for genes in panel_dict.values() for g in genes
                                     if g.upper() in lookup}), dtype=int)
    eligible = np.ones(n_genes, bool)
    eligible[all_panel_idx] = False

    def control_pool(b: int, min_size: int = 10) -> np.ndarray:
        # genes from the same expression bin, never panel genes; widen to adjacent bins if too few
        w = 0
        while True:
            pool = np.flatnonzero(eligible & (np.abs(bins - b) <= w))
            if len(pool) >= min_size or w >= n_bins:
                return pool
            w += 1

    pools = {b: control_pool(b) for b in range(n_bins)}

    cov_rows, effects, pvals = [], {}, {}
    for name, genes in panel_dict.items():
        hit = [g for g in genes if g.upper() in lookup]
        idx = [lookup[g.upper()] for g in hit]
        detected = [i for i in idx if gmean[i] > 0]
        usable = len(detected) >= min_genes
        cov_rows.append({"panel": name, "n_panel": len(genes), "n_in_data": len(hit), "n_detected": len(detected),
                         "detected": ";".join(syms[i] for i in detected), "usable": usable})
        if not usable:
            continue
        det = np.array(detected)
        k = len(det)
        obs = np.asarray(X[:, det].mean(axis=1)).ravel()
        exceed = np.zeros(X.shape[0])
        s1 = np.zeros(X.shape[0])
        s2 = np.zeros(X.shape[0])
        done = 0
        while done < n_null:  # null computed in column chunks to bound memory on large datasets
            m = min(250, n_null - done)
            ctrl = np.empty((m, k), dtype=int)
            for j, i in enumerate(det):
                pool = pools[bins[i]]
                ctrl[:, j] = pool[rng.integers(len(pool), size=m)]
            M = sparse.csr_matrix((np.full(m * k, 1.0 / k), (ctrl.ravel(), np.repeat(np.arange(m), k))),
                                  shape=(n_genes, m))
            null = (X @ M).toarray()
            exceed += (null >= obs[:, None]).sum(axis=1)
            s1 += null.sum(axis=1)
            s2 += (null ** 2).sum(axis=1)
            done += m
        mu = s1 / n_null
        sd = np.sqrt(np.maximum(s2 / n_null - mu ** 2, 0))
        effects[name] = (obs - mu) / np.where(sd > 1e-12, sd, np.inf)
        pvals[name] = (1 + exceed) / (n_null + 1)
    coverage = pd.DataFrame(cov_rows)
    return (pd.DataFrame(effects, index=adata.obs_names), pd.DataFrame(pvals, index=adata.obs_names), coverage)


def call_labels(effect: pd.DataFrame, pval: pd.DataFrame, alpha: float = 0.05,
                rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Label = significant panel with the largest effect; 'Unresolved' if none is significant.

    Significance uses a per-observation Bonferroni correction over the usable panels.
    """
    n = len(effect) if rows is None else int(rows.sum())
    if effect.shape[1] == 0:
        return np.repeat("Unresolved", n).astype(object), np.zeros(n)
    E = effect.to_numpy() if rows is None else effect.to_numpy()[rows]
    P = pval.to_numpy() if rows is None else pval.to_numpy()[rows]
    masked = np.where(P < alpha / max(E.shape[1], 1), E, -np.inf)
    best = np.argmax(masked, axis=1)
    top = masked[np.arange(len(masked)), best]
    srt = np.sort(masked, axis=1)
    second = srt[:, -2] if masked.shape[1] > 1 else np.full(len(masked), -np.inf)
    with np.errstate(invalid="ignore"):
        margin = np.where(np.isfinite(second), top - second, np.where(np.isfinite(top), top, 0.0))
    margin = np.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
    labels = effect.columns.to_numpy()[best].astype(object)
    labels[~np.isfinite(top)] = "Unresolved"
    return labels, margin


def annotate(adata, species: str = "auto", min_genes: int = 3, alpha: float = 0.05,
             layer: str | None = None, n_null: int = 1000, seed: int = 0) -> dict:
    """Annotate in place. Adds obs columns ``neuro_class``, ``neuro_subclass``,
    ``neuro_transmitter``, ``neuro_class_margin`` and ``state_*`` effects."""
    if species == "auto":
        species = detect_species(gene_symbols(adata))
    kw = dict(min_genes=min_genes, layer=layer, n_null=n_null, seed=seed)
    cls_e, cls_p, cls_cov = score_panels(adata, markers.panels("classes", species), **kw)
    cls, margin = call_labels(cls_e, cls_p, alpha)
    adata.obs["neuro_class"] = pd.Categorical(cls)
    adata.obs["neuro_class_margin"] = margin
    if cls_e.shape[1]:
        adata.obsm["neuro_class_effect"] = cls_e.to_numpy()
        adata.uns["neuro_class_effect_names"] = list(cls_e.columns)

    sub = cls.copy()
    sub_cov = []
    parent_of = markers.parents()
    sub_panels = markers.panels("neuron_subclasses", species)
    for parent in ("Excitatory neuron", "Inhibitory neuron"):
        m = cls == parent
        pnl = {k: v for k, v in sub_panels.items() if parent_of[k] == parent}
        e, p, cov = score_panels(adata, pnl, **kw)
        sub_cov.append(cov.assign(parent=parent))
        if m.sum() and e.shape[1]:
            lab, _ = call_labels(e, p, alpha, rows=m)
            lab = np.where(lab == "Unresolved", parent + " (subclass unresolved)", lab)
            sub[m] = lab
    adata.obs["neuro_subclass"] = pd.Categorical(sub)

    nt_e, nt_p, nt_cov = score_panels(adata, markers.panels("neurotransmitter", species), **kw)
    nt, _ = call_labels(nt_e, nt_p, alpha)
    neuronal = np.isin(cls, ["Excitatory neuron", "Inhibitory neuron"])
    adata.obs["neuro_transmitter"] = pd.Categorical(np.where(neuronal, nt, "Non-neuronal"))

    st_e, st_p, st_cov = score_panels(adata, markers.panels("states", species), **kw)
    for c in st_e.columns:
        key = c.split(" (")[0].replace(" / ", "_").replace(" ", "_").replace("-", "_").lower()
        adata.obs[f"state_{key}"] = st_e[c].to_numpy()

    summary = {
        "species": species,
        "marker_version": markers.version(),
        "alpha": alpha,
        "n_null": n_null,
        "usable_class_panels": cls_cov.loc[cls_cov["usable"], "panel"].tolist(),
        "unusable_class_panels": cls_cov.loc[~cls_cov["usable"], "panel"].tolist(),
        "class_counts": pd.Series(cls).value_counts().to_dict(),
        "subclass_counts": pd.Series(sub).value_counts().to_dict(),
        "transmitter_counts": adata.obs["neuro_transmitter"].value_counts().to_dict(),
        "fraction_unresolved": float(np.mean(cls == "Unresolved")),
    }
    if summary["unusable_class_panels"]:
        summary["warning"] = ("Some class panels are not covered by this gene list "
                              f"({', '.join(summary['unusable_class_panels'])}); those classes cannot be called.")
    adata.uns["neuro_annotation"] = summary
    coverage = pd.concat([cls_cov.assign(section="classes"), *[c.assign(section="neuron_subclasses") for c in sub_cov],
                          nt_cov.assign(section="neurotransmitter"), st_cov.assign(section="states")],
                         ignore_index=True)
    return {"summary": summary, "coverage": coverage, "class_effect": cls_e, "class_pvalue": cls_p}
