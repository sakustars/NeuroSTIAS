"""Sparse excitatory–inhibitory LIF network (Brunel 2000 model A, delta synapses).

NE excitatory and NI = NE/4 inhibitory neurons, each receiving C_E = εN_E
excitatory and C_I = εN_I inhibitory recurrent connections plus independent
Poisson external input at rate ν_ext = η·ν_thr. ``g`` is relative inhibition.
Regimes (synchronous-regular, asynchronous-irregular, ...) depend on (g, η).
Output: population rate, per-neuron rate and ISI CV, and a synchrony index
(variance of the population rate / mean single-neuron variance; Golomb 2007).

``scale_inhibition`` and ``scale_excitation`` support virtual perturbations
(e.g. reduced GABAergic drive).
"""

from __future__ import annotations

import numpy as np


def brunel_network(N_E: int = 2000, epsilon: float = 0.1, g: float = 5.0, eta: float = 2.0, J: float = 0.1,
                   D: float = 1.5, tau: float = 20.0, theta: float = 20.0, V_r: float = 10.0, t_ref: float = 2.0,
                   T: float = 1000.0, dt: float = 0.1, scale_inhibition: float = 1.0, scale_excitation: float = 1.0,
                   seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    N_I = N_E // 4
    N = N_E + N_I
    C_E, C_I = int(epsilon * N_E), int(epsilon * N_I)
    nu_thr = theta / (J * C_E * tau)            # kHz (per ms)
    nu_ext = eta * nu_thr
    # presynaptic -> postsynaptic adjacency as lists of targets per source
    pre_E = rng.integers(0, N_E, size=(N, C_E))
    pre_I = rng.integers(N_E, N, size=(N, C_I))
    targets = [[] for _ in range(N)]
    for post in range(N):
        for s in pre_E[post]:
            targets[s].append(post)
        for s in pre_I[post]:
            targets[s].append(post)
    targets = [np.asarray(t, dtype=int) for t in targets]
    w = np.where(np.arange(N) < N_E, J * scale_excitation, -g * J * scale_inhibition)
    steps = int(T / dt)
    dsteps = int(round(D / dt))
    ring = np.zeros((dsteps + 1, N))
    V = rng.uniform(0, theta, N)
    ref = np.zeros(N)
    lam_ext = nu_ext * C_E * dt
    spikes_t, spikes_i = [], []
    decay = np.exp(-dt / tau)
    for k in range(steps):
        slot = k % (dsteps + 1)
        inp = ring[slot].copy()
        ring[slot] = 0
        inp += J * rng.poisson(lam_ext, N)
        active = ref <= 0
        V[active] = V[active] * decay + inp[active]
        ref[~active] -= dt
        fired = np.flatnonzero(V >= theta)
        if fired.size:
            V[fired] = V_r
            ref[fired] = t_ref
            spikes_t.extend([k * dt] * fired.size)
            spikes_i.extend(fired.tolist())
            out = (k + dsteps) % (dsteps + 1)
            for s in fired:
                np.add.at(ring[out], targets[s], w[s])
    spikes_t = np.asarray(spikes_t)
    spikes_i = np.asarray(spikes_i)
    burn = 100.0
    m = spikes_t >= burn
    dur = (T - burn) / 1000.0
    rates = np.bincount(spikes_i[m], minlength=N) / dur
    cvs = []
    for i in range(0, N_E, max(1, N_E // 200)):
        st = spikes_t[m & (spikes_i == i)]
        if len(st) > 3:
            isi = np.diff(st)
            cvs.append(isi.std() / isi.mean())
    bins = np.arange(burn, T + 1, 1.0)
    pop = np.histogram(spikes_t[m], bins)[0] / (N * 1e-3)
    sample = rng.choice(N_E, size=min(200, N_E), replace=False)
    indiv = np.array([np.histogram(spikes_t[m & (spikes_i == i)], bins)[0] for i in sample])
    # Golomb (2007) chi^2: variance of the population-averaged signal over the mean single-neuron variance
    sync = float(indiv.mean(axis=0).var() / np.mean(indiv.var(axis=1))) if np.mean(indiv.var(axis=1)) > 0 else float("nan")
    return {"mean_rate_E_hz": float(rates[:N_E].mean()), "mean_rate_I_hz": float(rates[N_E:].mean()),
            "mean_isi_cv_E": float(np.mean(cvs)) if cvs else float("nan"), "synchrony_index": sync,
            "n_spikes": int(m.sum()), "nu_thr_hz": float(nu_thr * 1000), "params": dict(N_E=N_E, g=g, eta=eta, J=J, D=D,
            epsilon=epsilon, scale_inhibition=scale_inhibition, scale_excitation=scale_excitation, T=T, dt=dt, seed=seed),
            "population_rate_hz": pop}
