"""Cell-type prediction with a random forest (ported from V4.4.1 ``machine_learning_prediction_3d``).

Features are PCA of expression plus scaled coordinates. Evaluation uses a
grouped split (by section/sample when available; random otherwise, flagged), and
a training-label permutation control (labels shuffled, model retrained) gives an
empirical p-value. Spatially grouped validation is required before any claim of
generalisation (Roberts et al. 2017).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


def predict_cell_types(adata, label_key: str, group_key: str | None = None, n_estimators: int = 200,
                       max_depth: int | None = 12, test_size: float = 0.3, n_label_permutations: int = 20,
                       use_coordinates: bool = True, seed: int = 0) -> dict:
    y = adata.obs[label_key].astype(str).to_numpy()
    feats = [adata.obsm["X_pca"][:, :30]]
    if use_coordinates:
        c = np.asarray(adata.obsm["spatial_3d"] if "spatial_3d" in adata.obsm else adata.obsm["spatial"], dtype=float)
        feats.append(c)
    X = StandardScaler().fit_transform(np.hstack(feats))
    split = "random"
    if group_key and adata.obs[group_key].nunique() >= 2:
        groups = adata.obs[group_key].astype(str).to_numpy()
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y, groups))
        split = f"grouped by {group_key}"
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed,
                                  stratify=y if pd.Series(y).value_counts().min() >= 2 else None)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1,
                                class_weight="balanced")
    rf.fit(X[tr], y[tr])
    pred = rf.predict(X[te])
    acc = accuracy_score(y[te], pred)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_label_permutations):
        m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1)
        m.fit(X[tr], rng.permutation(y[tr]))
        null.append(accuracy_score(y[te], m.predict(X[te])))
    null = np.asarray(null)
    return {"split": split, "n_train": int(len(tr)), "n_test": int(len(te)), "accuracy": float(acc),
            "macro_f1": float(f1_score(y[te], pred, average="macro")),
            "null_accuracy_mean": float(null.mean()) if len(null) else None,
            "empirical_pvalue": float((1 + np.sum(null >= acc)) / (len(null) + 1)) if len(null) else None,
            "warning": None if split != "random" else "random split: spatial autocorrelation can inflate accuracy"}
