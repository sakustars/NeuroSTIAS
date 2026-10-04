"""LFP/EEG spectral analysis and EEG decoding (MNE is used where it is the reference tool)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}


def band_power(data: np.ndarray, sfreq: float, bands: dict = BANDS, nperseg_s: float = 2.0,
               channel_names: list[str] | None = None) -> pd.DataFrame:
    """Absolute and relative band power per channel (Welch PSD, trapezoidal integration).

    ``data`` is channels x samples.
    """
    data = np.atleast_2d(data)
    f, P = welch(data, fs=sfreq, nperseg=int(min(data.shape[1], nperseg_s * sfreq)))
    names = channel_names or [f"ch{i}" for i in range(data.shape[0])]
    fmax = min(max(b[1] for b in bands.values()), sfreq / 2)
    total_m = (f >= 1) & (f <= fmax)
    total = np.trapezoid(P[:, total_m], f[total_m], axis=1)
    rows = {}
    for b, (lo, hi) in bands.items():
        m = (f >= lo) & (f < min(hi, sfreq / 2))
        if m.sum() < 2:
            continue
        absb = np.trapezoid(P[:, m], f[m], axis=1)
        rows[f"{b}_abs"] = absb
        rows[f"{b}_rel"] = absb / total
    return pd.DataFrame(rows, index=names)


def motor_imagery_csp_lda(raw, events, event_id: dict, tmin: float = 1.0, tmax: float = 2.0,
                          l_freq: float = 7.0, h_freq: float = 30.0, n_components: int = 4, n_splits: int = 10,
                          seed: int = 42) -> dict:
    """Band-pass → epochs → CSP + LDA with shuffle-split cross-validation (as in the MNE reference example)."""
    import mne
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import ShuffleSplit, cross_val_score
    from sklearn.pipeline import Pipeline

    raw = raw.copy().filter(l_freq, h_freq, fir_design="firwin", skip_by_annotation="edge", verbose="ERROR")
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    epochs = mne.Epochs(raw, events, event_id, -1.0, 4.0, proj=True, picks=picks, baseline=None, preload=True,
                        verbose="ERROR")
    ep = epochs.copy().crop(tmin=tmin, tmax=tmax)
    X = ep.get_data(copy=False)
    y = ep.events[:, -1] - min(event_id.values())
    cv = ShuffleSplit(n_splits, test_size=0.2, random_state=seed)
    clf = Pipeline([("CSP", CSP(n_components=n_components, reg=None, log=True, norm_trace=False)),
                    ("LDA", LinearDiscriminantAnalysis())])
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=None)
    chance = max(np.mean(y == y[0]), 1 - np.mean(y == y[0]))
    return {"accuracy_mean": float(scores.mean()), "accuracy_sd": float(scores.std()), "chance": float(chance),
            "n_epochs": int(len(y)), "scores": scores.tolist()}
