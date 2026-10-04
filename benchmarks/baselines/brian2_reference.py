"""Brian2 reference simulations (Stimberg et al. 2019) for E09, run in benchmarks/envs/brian2.

Usage: python brian2_reference.py --task {hh_trace,hh_fi,lif_fi,brunel} --out out.json [--seed 0]
Parameters mirror neurostias.modeling exactly (see E09 run.py).
"""
import argparse
import json
import time

import numpy as np
from brian2 import (NeuronGroup, PoissonInput, SpikeMonitor, StateMonitor, Synapses, defaultclock, ms, mV, nA, Mohm,
                    prefs, run, seed, start_scope, uF, cm, msiemens, uA, Network)

prefs.codegen.target = "numpy"

HH_EQS = """
dv/dt = (I - gna*m**3*h*(v-ENa) - gk*n**4*(v-EK) - gl*(v-El))/Cm : volt
dm/dt = alpham*(1-m) - betam*m : 1
dh/dt = alphah*(1-h) - betah*h : 1
dn/dt = alphan*(1-n) - betan*n : 1
alpham = 0.1/mV*(v+40*mV)/(1-exp(-(v+40*mV)/(10*mV)))/ms : Hz
betam = 4*exp(-(v+65*mV)/(18*mV))/ms : Hz
alphah = 0.07*exp(-(v+65*mV)/(20*mV))/ms : Hz
betah = 1/(1+exp(-(v+35*mV)/(10*mV)))/ms : Hz
alphan = 0.01/mV*(v+55*mV)/(1-exp(-(v+55*mV)/(10*mV)))/ms : Hz
betan = 0.125*exp(-(v+65*mV)/(80*mV))/ms : Hz
I : amp/meter**2
"""
NS = dict(Cm=1 * uF / cm ** 2, gna=120 * msiemens / cm ** 2, gk=36 * msiemens / cm ** 2, gl=0.3 * msiemens / cm ** 2,
          ENa=50 * mV, EK=-77 * mV, El=-54.387 * mV)


def hh_group(n, I_vals, dt):
    defaultclock.dt = dt * ms
    G = NeuronGroup(n, HH_EQS, method="rk4", threshold="v > 0*mV", refractory="v > 0*mV", namespace=NS)
    G.v = -65 * mV
    v = -65.0
    an = 0.01 * (v + 55) / (1 - np.exp(-(v + 55) / 10)); bn = 0.125 * np.exp(-(v + 65) / 80)
    am = 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10)); bm = 4 * np.exp(-(v + 65) / 18)
    ah = 0.07 * np.exp(-(v + 65) / 20); bh = 1 / (1 + np.exp(-(v + 35) / 10))
    G.n, G.m, G.h = an / (an + bn), am / (am + bm), ah / (ah + bh)
    G.I = np.asarray(I_vals) * uA / cm ** 2
    return G


def task_hh_trace(a):
    start_scope()
    G = hh_group(1, [10.0], a.dt)
    M = StateMonitor(G, "v", record=0)
    t0 = time.time()
    run(100 * ms)
    return {"t": (M.t / ms).tolist(), "v": (M.v[0] / mV).tolist(), "seconds": time.time() - t0}


def task_hh_fi(a):
    start_scope()
    I = [float(x) for x in a.currents.split(",")]
    G = hh_group(len(I), I, a.dt)
    S = SpikeMonitor(G)
    t0 = time.time()
    run(500 * ms)
    st = S.spike_trains()
    rates = [1000.0 * np.sum(np.asarray(st[i] / ms) >= 100) / 400.0 for i in range(len(I))]
    return {"currents": I, "rates": rates, "seconds": time.time() - t0}


def task_lif_fi(a):
    start_scope()
    defaultclock.dt = a.dt * ms
    I = [float(x) for x in a.currents.split(",")]
    eqs = "dv/dt = (-(v - El) + R*I)/tau : volt (unless refractory)\nI : amp"
    G = NeuronGroup(len(I), eqs, threshold="v >= -50*mV", reset="v = -70*mV", refractory=2 * ms, method="exact",
                    namespace=dict(El=-70 * mV, R=10 * Mohm, tau=20 * ms))
    G.v = -70 * mV
    G.I = np.asarray(I) * nA
    S = SpikeMonitor(G)
    run(2000 * ms)
    return {"currents": I, "rates": (np.asarray(S.count) / 2.0).tolist()}


def task_brunel(a):
    start_scope()
    seed(a.seed)
    defaultclock.dt = a.dt * ms
    NE, g, eta, J, D, tau, theta, Vr = 2000, a.g, a.eta, 0.1 * mV, 1.5 * ms, 20 * ms, 20 * mV, 10 * mV
    NI = NE // 4
    CE, CI = int(0.1 * NE), int(0.1 * NI)
    nu_thr = theta / (J * CE * tau)
    G = NeuronGroup(NE + NI, "dv/dt = -v/tau : volt (unless refractory)", threshold="v >= theta", reset="v = Vr",
                    refractory=2 * ms, method="exact", namespace=dict(tau=tau, theta=theta, Vr=Vr))
    G.v = "rand()*theta"
    E, I = G[:NE], G[NE:]
    # fixed in-degree as in Brunel (2000) and neurostias: each neuron receives exactly CE excitatory and
    # CI inhibitory inputs drawn with replacement
    rng = np.random.default_rng(a.seed)
    post = np.repeat(np.arange(NE + NI), CE)
    SE = Synapses(E, G, on_pre="v_post += J", delay=D, namespace=dict(J=J))
    SE.connect(i=rng.integers(0, NE, size=len(post)), j=post)
    posti = np.repeat(np.arange(NE + NI), CI)
    SI = Synapses(I, G, on_pre="v_post += -g*J", delay=D, namespace=dict(J=J, g=g))
    SI.connect(i=rng.integers(0, NI, size=len(posti)), j=posti)
    P = PoissonInput(G, "v", CE, eta * nu_thr, weight=J)
    M = SpikeMonitor(G)
    t0 = time.time()
    run(1000 * ms)
    t = np.asarray(M.t / ms)
    i = np.asarray(M.i)
    m = t >= 100
    rates = np.bincount(i[m], minlength=NE + NI) / 0.9
    cvs = []
    for k in range(0, NE, NE // 200):
        s = t[m & (i == k)]
        if len(s) > 3:
            d = np.diff(s)
            cvs.append(d.std() / d.mean())
    bins = np.arange(100, 1001, 1.0)
    rng = np.random.default_rng(a.seed)
    smp = rng.choice(NE, 200, replace=False)
    indiv = np.array([np.histogram(t[m & (i == k)], bins)[0] for k in smp])
    sync = float(indiv.mean(0).var() / np.mean(indiv.var(1))) if np.mean(indiv.var(1)) > 0 else float("nan")
    return {"mean_rate_E_hz": float(rates[:NE].mean()), "mean_rate_I_hz": float(rates[NE:].mean()),
            "mean_isi_cv_E": float(np.mean(cvs)) if cvs else float("nan"), "synchrony_index": sync,
            "seconds": time.time() - t0}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--currents", default="2,4,6,6.5,7,8,10,15,20,30")
    p.add_argument("--g", type=float, default=5.0)
    p.add_argument("--eta", type=float, default=2.0)
    a = p.parse_args()
    res = {"hh_trace": task_hh_trace, "hh_fi": task_hh_fi, "lif_fi": task_lif_fi, "brunel": task_brunel}[a.task](a)
    json.dump(res, open(a.out, "w"))
