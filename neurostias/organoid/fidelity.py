"""Brain-organoid fidelity, region identity and maturation against a primary reference.

Given organoid single-cell data (query) and a labelled primary developing-brain
reference, this module provides:

* **label transfer** - kNN in a PCA space fitted on the reference (query projected,
  never refitted), with a per-cell confidence = fraction of neighbours agreeing;
* **region identity** - Spearman correlation of each query cluster's mean profile
  with reference region profiles on reference-informative genes (the approach of
  VoxHunt, Fleck et al. 2021, applied to a single-cell reference);
* **transcriptomic similarity ("fidelity")** - for each query cell, the
  correlation to its nearest reference centroid of the transferred type, divided by
  the median correlation of reference cells of that type to the same centroid
  (1.0 = as similar as primary cells are to themselves);
* **maturation** - neurodevelopmental stage enrichment (``legacy.development``) and,
  if the reference has ages, a kNN-regressed reference age per cell;
* **cell stress** - glycolysis / ER-stress programme reported to be elevated in
  organoids (Bhaduri et al. 2020 Nature: PGK1, ARCN1, GORASP2 and related genes).

No result is produced for a component whose inputs are missing; the status says why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

STRESS_GENES = {
    "Glycolysis stress": ["PGK1", "ALDOA", "ENO1", "LDHA", "PGAM1", "TPI1", "BNIP3"],
    "ER stress": ["ARCN1", "GORASP2", "DDIT3", "ATF4", "XBP1", "HSPA5", "HERPUD1"],
}


def _lognorm(adata):
    X = adata.layers["log_normalized"] if "log_normalized" in adata.layers else adata.X
    return X


def _shared(query, ref):
    qs = pd.Index(query.var_names.astype(str).str.upper())
    rs = pd.Index(ref.var_names.astype(str).str.upper())
    common = qs.intersection(rs)
    common = common[~common.duplicated()]
    qi = qs.get_indexer(common)
    ri = rs.get_indexer(common)
    return common, qi, ri


def _dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)


def informative_genes(ref, label_key: str, n_per_label: int = 50) -> list[str]:
    tmp = ref.copy()
    if "log_normalized" in tmp.layers:
        tmp.X = tmp.layers["log_normalized"]
    sc.tl.rank_genes_groups(tmp, label_key, method="t-test", n_genes=n_per_label)
    names = pd.DataFrame(tmp.uns["rank_genes_groups"]["names"])
    return sorted({str(g).upper() for g in names.to_numpy().ravel()})


def map_to_reference(query, ref, label_key: str, n_pcs: int = 30, k: int = 15, n_hvg: int = 2000,
                     age_key: str | None = None, seed: int = 0) -> dict:
    common, qi, ri = _shared(query, ref)
    if len(common) < 200:
        return {"status": "not_applicable", "reason": f"only {len(common)} shared genes"}
    R = _dense(_lognorm(ref)[:, ri])
    Q = _dense(_lognorm(query)[:, qi])
    var = R.var(axis=0)
    hv = np.argsort(var)[::-1][: min(n_hvg, len(common))]
    mu, sd = R[:, hv].mean(0), R[:, hv].std(0)
    sd[sd == 0] = 1
    pca = PCA(n_components=min(n_pcs, len(hv) - 1), random_state=seed).fit((R[:, hv] - mu) / sd)
    Rp = pca.transform((R[:, hv] - mu) / sd)
    Qp = pca.transform((Q[:, hv] - mu) / sd)
    nn = NearestNeighbors(n_neighbors=k).fit(Rp)
    _, idx = nn.kneighbors(Qp)
    ref_labels = ref.obs[label_key].astype(str).to_numpy()
    votes = ref_labels[idx]
    pred, conf = [], []
    for row in votes:
        u, c = np.unique(row, return_counts=True)
        pred.append(u[np.argmax(c)])
        conf.append(c.max() / k)
    query.obs["ref_label"] = pd.Categorical(pred)
    query.obs["ref_label_confidence"] = conf
    out = {"status": "completed", "n_shared_genes": int(len(common)), "label_counts": pd.Series(pred).value_counts().to_dict(),
           "mean_confidence": float(np.mean(conf))}
    if age_key and age_key in ref.obs:
        ages = pd.to_numeric(ref.obs[age_key], errors="coerce").to_numpy()
        query.obs["ref_age_knn"] = np.nanmedian(ages[idx], axis=1)
        out["median_ref_age"] = float(np.nanmedian(query.obs["ref_age_knn"]))
    # fidelity: correlation to transferred-type centroid relative to reference self-similarity
    cent = pd.DataFrame(R[:, hv]).groupby(ref_labels).mean()
    fid = np.full(query.n_obs, np.nan)
    pred = np.asarray(pred)
    for lab in np.unique(pred):
        if lab not in cent.index:
            continue
        c = cent.loc[lab].to_numpy()
        rm = ref_labels == lab
        rc = np.array([np.corrcoef(x, c)[0, 1] for x in R[rm][:, hv]])
        base = np.nanmedian(rc) if np.isfinite(rc).any() else np.nan
        qm = pred == lab
        qc = np.array([np.corrcoef(x, c)[0, 1] for x in Q[qm][:, hv]])
        fid[qm] = qc / base if base and np.isfinite(base) else np.nan
    query.obs["fidelity"] = fid
    out["median_fidelity"] = float(np.nanmedian(fid))
    return out


def region_identity(query, ref, region_key: str, cluster_key: str, genes: list[str] | None = None) -> pd.DataFrame:
    """Spearman correlation of query cluster means vs reference region means (VoxHunt-style)."""
    common, qi, ri = _shared(query, ref)
    if genes is not None:
        keep = np.isin(common, [g.upper() for g in genes])
        common, qi, ri = common[keep], qi[keep], ri[keep]
    R = pd.DataFrame(_dense(_lognorm(ref)[:, ri]), columns=common).groupby(ref.obs[region_key].astype(str).to_numpy()).mean()
    Q = pd.DataFrame(_dense(_lognorm(query)[:, qi]), columns=common).groupby(query.obs[cluster_key].astype(str).to_numpy()).mean()
    out = pd.DataFrame(index=Q.index, columns=R.index, dtype=float)
    for q in Q.index:
        for r in R.index:
            out.loc[q, r] = spearmanr(Q.loc[q], R.loc[r])[0]
    return out


def stress_scores(adata, n_null: int = 1000, seed: int = 0) -> pd.DataFrame:
    from ..atlas.annotate import score_panels
    eff, pv, cov = score_panels(adata, STRESS_GENES, min_genes=3, n_null=n_null, seed=seed)
    for c in eff.columns:
        adata.obs[f"stress_{c.split()[0].lower()}"] = eff[c].to_numpy()
    return cov
