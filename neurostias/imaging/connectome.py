"""Functional connectomes from regional fMRI time series, graph metrics and group classification.

Connectivity estimation uses nilearn's ``ConnectivityMeasure`` (correlation,
partial correlation or tangent space; Dadi et al. 2019 recommend tangent for
prediction). Graph metrics are computed with networkx on a proportionally
thresholded absolute-weight graph. Classification uses stratified k-fold CV;
when several scans come from one subject, use grouped CV.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def connectivity(time_series: list[np.ndarray], kind: str = "tangent") -> np.ndarray:
    from nilearn.connectome import ConnectivityMeasure
    cm = ConnectivityMeasure(kind=kind, vectorize=False, discard_diagonal=False)
    return cm.fit_transform(time_series)


def vectorize(mats: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(mats.shape[1], k=1)
    return np.array([m[iu] for m in mats])


def graph_metrics(mat: np.ndarray, density: float = 0.15) -> dict:
    A = np.abs(np.asarray(mat, dtype=float))
    np.fill_diagonal(A, 0)
    iu = np.triu_indices(len(A), 1)
    thr = np.quantile(A[iu], 1 - density)
    B = np.where(A >= thr, A, 0.0)
    G = nx.from_numpy_array(B)
    comms = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    deg = np.array([d for _, d in G.degree()])
    return {"density": float(nx.density(G)), "mean_degree": float(deg.mean()),
            "clustering": float(nx.average_clustering(G, weight="weight")),
            "global_efficiency": float(nx.global_efficiency(G)),
            "modularity": float(nx.algorithms.community.modularity(G, comms, weight="weight")),
            "n_modules": int(len(comms))}


def classify(features: np.ndarray, labels, groups=None, n_splits: int = 5, seed: int = 0) -> dict:
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    y = np.asarray(labels)
    clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=20000, random_state=seed))
    cv = GroupKFold(n_splits) if groups is not None else StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, features, y, cv=cv, groups=groups)
    chance = pd.Series(y).value_counts(normalize=True).max()
    return {"accuracy_mean": float(scores.mean()), "accuracy_sd": float(scores.std()), "chance": float(chance),
            "scores": scores.tolist(), "n": int(len(y))}
