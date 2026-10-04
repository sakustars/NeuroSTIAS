"""Single-neuron models: Hodgkin–Huxley (1952 squid axon) and leaky integrate-and-fire.

Units: mV, ms, µA/cm², mS/cm², µF/cm². HH parameters are the classic values
shifted to a −65 mV rest (as in Dayan & Abbott 2001 and the Brian2 examples).
Integration is classical RK4 with step ``dt`` (default 0.01 ms). Experiment E09
compares these against Brian2 and, for LIF, against the closed-form f–I curve.

Pharmacological perturbations scale maximal conductances: e.g. ``g_na_scale=0``
mimics complete tetrodotoxin (TTX) block, ``g_k_scale<1`` partial TEA block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class HHParams:
    C_m: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3
    E_na: float = 50.0
    E_k: float = -77.0
    E_l: float = -54.387
    g_na_scale: float = 1.0
    g_k_scale: float = 1.0


def _rates(V):
    with np.errstate(divide="ignore", invalid="ignore"):
        an = np.where(np.abs(V + 55) < 1e-7, 0.1, 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10)))
        am = np.where(np.abs(V + 40) < 1e-7, 1.0, 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10)))
    bn = 0.125 * np.exp(-(V + 65) / 80)
    bm = 4.0 * np.exp(-(V + 65) / 18)
    ah = 0.07 * np.exp(-(V + 65) / 20)
    bh = 1.0 / (1 + np.exp(-(V + 35) / 10))
    return an, bn, am, bm, ah, bh


def _deriv(y, I, p: HHParams):
    V, n, m, h = y
    an, bn, am, bm, ah, bh = _rates(V)
    I_na = p.g_na * p.g_na_scale * m ** 3 * h * (V - p.E_na)
    I_k = p.g_k * p.g_k_scale * n ** 4 * (V - p.E_k)
    I_l = p.g_l * (V - p.E_l)
    return np.array([(I - I_na - I_k - I_l) / p.C_m, an * (1 - n) - bn * n, am * (1 - m) - bm * m,
                     ah * (1 - h) - bh * h])


def simulate_hh(I_ext, T: float = 200.0, dt: float = 0.01, params: HHParams | None = None, V0: float = -65.0) -> dict:
    """I_ext: constant (float) or callable t -> current. Returns time, voltage and spike times (0-mV upcrossings)."""
    p = params or HHParams()
    an, bn, am, bm, ah, bh = _rates(np.array(V0))
    y = np.array([V0, an / (an + bn), am / (am + bm), ah / (ah + bh)], dtype=float)
    n = int(round(T / dt))
    t = np.arange(n + 1) * dt
    V = np.empty(n + 1)
    V[0] = V0
    f = I_ext if callable(I_ext) else (lambda _t: I_ext)
    for i in range(n):
        ti = t[i]
        k1 = _deriv(y, f(ti), p)
        k2 = _deriv(y + 0.5 * dt * k1, f(ti + dt / 2), p)
        k3 = _deriv(y + 0.5 * dt * k2, f(ti + dt / 2), p)
        k4 = _deriv(y + dt * k3, f(ti + dt), p)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        V[i + 1] = y[0]
    up = np.flatnonzero((V[:-1] < 0) & (V[1:] >= 0))
    spikes = t[up] + dt * (-V[up]) / (V[up + 1] - V[up])
    return {"t": t, "V": V, "spikes": spikes, "params": asdict(p)}


def hh_fi_curve(currents, T: float = 500.0, discard: float = 100.0, dt: float = 0.01, params: HHParams | None = None):
    rates = []
    for I in currents:
        s = simulate_hh(float(I), T=T, dt=dt, params=params)["spikes"]
        s = s[s >= discard]
        rates.append(1000.0 * len(s) / (T - discard))
    return np.asarray(rates)


@dataclass
class LIFParams:
    tau_m: float = 20.0     # ms
    E_l: float = -70.0      # mV
    V_th: float = -50.0
    V_reset: float = -70.0
    R_m: float = 10.0       # MΩ (I in nA -> R*I in mV)
    t_ref: float = 2.0      # ms


def simulate_lif(I_ext, T: float = 1000.0, dt: float = 0.01, params: LIFParams | None = None) -> dict:
    """Exact exponential integration between steps for constant input within a step."""
    p = params or LIFParams()
    n = int(round(T / dt))
    t = np.arange(n + 1) * dt
    V = np.empty(n + 1)
    V[0] = p.E_l
    f = I_ext if callable(I_ext) else (lambda _t: I_ext)
    decay = np.exp(-dt / p.tau_m)
    spikes, ref_until = [], -1.0
    for i in range(n):
        if t[i] < ref_until:
            V[i + 1] = p.V_reset
            continue
        Vinf = p.E_l + p.R_m * f(t[i])
        V[i + 1] = Vinf + (V[i] - Vinf) * decay
        if V[i + 1] >= p.V_th:
            # linear interpolation of the crossing time
            frac = (p.V_th - V[i]) / (V[i + 1] - V[i])
            ts = t[i] + frac * dt
            spikes.append(ts)
            V[i + 1] = p.V_reset
            ref_until = ts + p.t_ref
    return {"t": t, "V": V, "spikes": np.asarray(spikes), "params": asdict(p)}


def lif_rate_analytic(I, params: LIFParams | None = None) -> np.ndarray:
    """Closed-form LIF firing rate (Hz) for constant current I (nA)."""
    p = params or LIFParams()
    I = np.atleast_1d(np.asarray(I, dtype=float))
    drive = p.R_m * I
    thr = p.V_th - p.E_l
    out = np.zeros_like(I)
    m = drive > thr
    isi = p.t_ref + p.tau_m * np.log((drive[m] - (p.V_reset - p.E_l)) / (drive[m] - thr))
    out[m] = 1000.0 / isi
    return out


def lif_fi_curve(currents, T: float = 2000.0, dt: float = 0.01, params: LIFParams | None = None):
    return np.array([1000.0 * len(simulate_lif(float(I), T=T, dt=dt, params=params)["spikes"]) / T for I in currents])
