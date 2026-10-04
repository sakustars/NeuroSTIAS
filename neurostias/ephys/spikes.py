"""Spike-train analysis: loading (NWB units table), firing statistics, QC and PSTHs.

Spike times are in seconds. Definitions follow common practice (Elephant / Allen
Institute ecephys QC); experiment E08 checks agreement with Elephant.

* firing rate = n_spikes / duration
* ISI CV = SD(ISI) / mean(ISI)            (Poisson ≈ 1)
* local variation LV (Shinomoto et al. 2003) = 3/(n-1) Σ ((I_i - I_{i+1}) / (I_i + I_{i+1}))²
* refractory-period violation fraction = fraction of ISIs < ``refractory_ms``
* burst fraction = fraction of spikes in ISIs < ``burst_isi_ms``
* Fano factor of spike counts in windows of ``fano_window`` seconds
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_nwb_units(path: str | Path, max_units: int | None = None) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Return ({unit_id: spike_times}, units metadata) from an NWB file's units table."""
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        if nwb.units is None:
            raise ValueError("NWB file has no units table")
        df = nwb.units.to_dataframe()
        ids = list(df.index)[: max_units or None]
        trains = {str(u): np.asarray(df.loc[u, "spike_times"], dtype=float) for u in ids}
        meta = df.loc[ids].drop(columns=["spike_times"], errors="ignore")
        meta = meta[[c for c in meta.columns if meta[c].map(np.isscalar).all()]]
    return trains, meta


def unit_metrics(st: np.ndarray, t_start: float | None = None, t_stop: float | None = None,
                 refractory_ms: float = 1.5, burst_isi_ms: float = 10.0, fano_window: float = 1.0) -> dict:
    st = np.sort(np.asarray(st, dtype=float))
    t0 = st.min() if t_start is None and len(st) else (t_start or 0.0)
    t1 = st.max() if t_stop is None and len(st) else (t_stop or 0.0)
    dur = max(t1 - t0, 1e-12)
    isi = np.diff(st)
    out = {"n_spikes": int(len(st)), "duration_s": float(dur), "firing_rate_hz": float(len(st) / dur)}
    if len(isi) >= 2:
        out["isi_cv"] = float(isi.std() / isi.mean())
        r = (isi[:-1] - isi[1:]) / (isi[:-1] + isi[1:])
        out["lv"] = float(3.0 / (len(isi) - 1) * np.sum(r ** 2))
    else:
        out["isi_cv"] = out["lv"] = float("nan")
    out["refractory_violation_fraction"] = float(np.mean(isi < refractory_ms / 1000)) if len(isi) else float("nan")
    if len(isi):
        short = isi < burst_isi_ms / 1000
        in_burst = np.zeros(len(st), bool)
        in_burst[:-1] |= short
        in_burst[1:] |= short
        out["burst_spike_fraction"] = float(in_burst.mean())
    else:
        out["burst_spike_fraction"] = float("nan")
    edges = np.arange(t0, t1 + fano_window, fano_window)
    if len(edges) > 2:
        counts = np.histogram(st, edges)[0]
        out["fano_factor"] = float(counts.var() / counts.mean()) if counts.mean() > 0 else float("nan")
    else:
        out["fano_factor"] = float("nan")
    return out


def population_metrics(trains: dict[str, np.ndarray], **kw) -> pd.DataFrame:
    t0 = min((t.min() for t in trains.values() if len(t)), default=0.0)
    t1 = max((t.max() for t in trains.values() if len(t)), default=0.0)
    rows = [{"unit": u, **unit_metrics(t, t0, t1, **kw)} for u, t in trains.items()]
    return pd.DataFrame(rows).set_index("unit")


def psth(st: np.ndarray, events: np.ndarray, window: tuple[float, float] = (-0.5, 1.0), bin_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(window[0], window[1] + bin_s / 2, bin_s)
    counts = np.zeros(len(edges) - 1)
    st = np.sort(st)
    for e in events:
        lo, hi = np.searchsorted(st, [e + window[0], e + window[1]])
        counts += np.histogram(st[lo:hi] - e, edges)[0]
    return edges[:-1] + bin_s / 2, counts / (len(events) * bin_s)


def cross_correlogram(a: np.ndarray, b: np.ndarray, window_s: float = 0.05, bin_s: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(-window_s, window_s + bin_s / 2, bin_s)
    b = np.sort(b)
    counts = np.zeros(len(edges) - 1)
    for t in a:
        lo, hi = np.searchsorted(b, [t - window_s, t + window_s])
        counts += np.histogram(b[lo:hi] - t, edges)[0]
    return edges[:-1] + bin_s / 2, counts
