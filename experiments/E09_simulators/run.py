"""E09 - Correctness of the NeuroSTIAS neuron and network simulators (validation experiment).

1. Hodgkin-Huxley (classic 1952 parameters, -65 mV rest, RK4, dt = 0.01 ms): voltage trace at
   10 uA/cm2 vs Brian2 (rk4) - RMSE and spike-time differences; f-I curve vs Brian2.
2. LIF: simulated f-I vs the closed-form rate and vs Brian2 (exact integration).
3. Brunel (2000) sparse E-I network, (g, eta) in {(5, 2), (4.5, 0.9), (6, 4)}, 5 seeds each:
   mean E rate, ISI CV and synchrony vs a Brian2 implementation with the same parameters.
   Networks are random, so agreement is statistical (means over seeds), not spike-exact.
"""
import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neurostias.experiment import Experiment, md_table  # noqa: E402
from neurostias.modeling.network import brunel_network  # noqa: E402
from neurostias.modeling.neurons import hh_fi_curve, lif_fi_curve, lif_rate_analytic, simulate_hh  # noqa: E402

BR = ROOT / "benchmarks" / "envs" / "brian2" / "bin" / "python"
SCRIPT = ROOT / "benchmarks" / "baselines" / "brian2_reference.py"
CURRENTS = [2, 4, 6, 6.5, 7, 8, 10, 15, 20, 30]


def brian(task, **kw):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "o.json"
        args = [str(BR), str(SCRIPT), "--task", task, "--out", str(out)] + sum(([f"--{k}", str(v)] for k, v in kw.items()), [])
        r = subprocess.run(args, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-2000:])
        return json.loads(out.read_text())


def main():
    exp = Experiment("E09", "Neuron and network simulator correctness", "Do the NeuroSTIAS Hodgkin-Huxley, LIF and Brunel "
                     "network simulators agree with Brian2 and with closed-form results?", data_origin="synthetic",
                     params={"hh_dt_ms": 0.01, "currents": CURRENTS, "brunel_regimes": [(5, 2), (4.5, 0.9), (6, 4)]})
    # 1. HH trace
    t0 = time.time()
    ours = simulate_hh(10.0, T=100.0, dt=0.01)
    t_ours = time.time() - t0
    try:
        b = brian("hh_trace", dt=0.01)
        bv = np.interp(ours["t"], b["t"], b["v"])
        rmse = float(np.sqrt(np.mean((ours["V"] - bv) ** 2)))
        bt = np.asarray(b["t"]); bvv = np.asarray(b["v"])
        up = np.flatnonzero((bvv[:-1] < 0) & (bvv[1:] >= 0))
        b_sp = bt[up]
        n = min(len(b_sp), len(ours["spikes"]))
        dspk = float(np.max(np.abs(b_sp[:n] - ours["spikes"][:n]))) if n else float("nan")
        hh_trace = {"rmse_mV": rmse, "n_spikes_ours": int(len(ours["spikes"])), "n_spikes_brian2": int(len(b_sp)),
                    "max_spike_time_diff_ms": dspk, "seconds_ours": t_ours, "seconds_brian2": b["seconds"]}
        exp.baseline("brian2", "ran")
    except Exception as exc:  # noqa: BLE001
        hh_trace = {"error": str(exc)[:500]}
        exp.baseline("brian2", "failed", str(exc))
    # HH f-I
    fi_ours = hh_fi_curve(CURRENTS, T=500.0, discard=100.0, dt=0.01)
    fi = pd.DataFrame({"current_uA_cm2": CURRENTS, "rate_ours_hz": fi_ours})
    try:
        fb = brian("hh_fi", dt=0.01, currents=",".join(map(str, CURRENTS)))
        fi["rate_brian2_hz"] = fb["rates"]
    except Exception as exc:  # noqa: BLE001
        exp.note(f"Brian2 HH f-I failed: {exc}")
    exp.table(fi, "e09_hh_fi.csv")
    # 2. LIF
    I = [1.9, 2.1, 2.5, 3.0, 4.0, 6.0]
    lif = pd.DataFrame({"current_nA": I, "rate_ours_hz": lif_fi_curve(I, T=2000.0), "rate_analytic_hz": lif_rate_analytic(I)})
    try:
        lif["rate_brian2_hz"] = brian("lif_fi", dt=0.01, currents=",".join(map(str, I)))["rates"]
    except Exception as exc:  # noqa: BLE001
        exp.note(f"Brian2 LIF failed: {exc}")
    exp.table(lif, "e09_lif_fi.csv")
    # 3. Brunel
    rows = []
    for g, eta in ((5.0, 2.0), (4.5, 0.9), (6.0, 4.0)):
        for s in exp.seeds:
            t0 = time.time()
            o = brunel_network(N_E=2000, g=g, eta=eta, T=1000.0, seed=s)
            rows.append({"g": g, "eta": eta, "seed": s, "simulator": "neurostias", "rate_E": o["mean_rate_E_hz"],
                         "cv_E": o["mean_isi_cv_E"], "synchrony": o["synchrony_index"], "seconds": time.time() - t0})
            try:
                b = brian("brunel", seed=s, g=g, eta=eta, dt=0.1)
                rows.append({"g": g, "eta": eta, "seed": s, "simulator": "brian2", "rate_E": b["mean_rate_E_hz"],
                             "cv_E": b["mean_isi_cv_E"], "synchrony": b["synchrony_index"], "seconds": b["seconds"]})
            except Exception as exc:  # noqa: BLE001
                exp.note(f"Brian2 Brunel failed ({g},{eta},{s}): {str(exc)[:200]}")
            print("brunel", g, eta, s, flush=True)
    # 4. time-step convergence in the fluctuation-driven regime (g=4.5, eta=0.9), where within-step event
    #    ordering differs between simulators (Brian2 applies synaptic input after threshold detection)
    conv = []
    for dt in (0.1, 0.05, 0.025):
        for s in exp.seeds[:3]:
            o = brunel_network(N_E=2000, g=4.5, eta=0.9, T=1000.0, dt=dt, seed=s)
            conv.append({"dt_ms": dt, "seed": s, "simulator": "neurostias", "rate_E": o["mean_rate_E_hz"]})
            try:
                b = brian("brunel", seed=s, g=4.5, eta=0.9, dt=dt)
                conv.append({"dt_ms": dt, "seed": s, "simulator": "brian2", "rate_E": b["mean_rate_E_hz"]})
            except Exception as exc:  # noqa: BLE001
                exp.note(f"Brian2 convergence run failed: {str(exc)[:200]}")
            print("convergence", dt, s, flush=True)
    conv = pd.DataFrame(conv)
    exp.table(conv, "e09_brunel_dt_convergence_raw.csv")
    conv_s = conv.groupby(["dt_ms", "simulator"], as_index=False)["rate_E"].mean().pivot(index="dt_ms", columns="simulator",
                                                                                           values="rate_E").reset_index()
    if {"brian2", "neurostias"} <= set(conv_s.columns):
        conv_s["relative_difference"] = (conv_s["neurostias"] - conv_s["brian2"]) / conv_s["brian2"]
    exp.table(conv_s, "e09_brunel_dt_convergence.csv")
    exp.summary["dt_convergence"] = conv_s.to_dict("records")
    br = pd.DataFrame(rows)
    exp.table(br, "e09_brunel_raw.csv")
    bsum = br.groupby(["g", "eta", "simulator"], as_index=False).agg(rate_E_mean=("rate_E", "mean"), rate_E_sd=("rate_E", "std"),
                                                                     cv_E_mean=("cv_E", "mean"), synchrony_mean=("synchrony", "mean"),
                                                                     seconds_median=("seconds", "median"))
    exp.table(bsum, "e09_brunel_summary.csv")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    axes[0].plot(ours["t"], ours["V"], "k", lw=0.8, label="NeuroSTIAS")
    if "rate_brian2_hz" in fi:
        axes[0].plot(ours["t"], bv, "--", color="#D55E00", lw=0.8, label="Brian2")
    axes[0].set_xlabel("t (ms)"); axes[0].set_ylabel("V (mV)"); axes[0].legend(frameon=False); axes[0].set_title("HH, 10 uA/cm2")
    axes[1].plot(fi.current_uA_cm2, fi.rate_ours_hz, "o-", label="NeuroSTIAS")
    if "rate_brian2_hz" in fi:
        axes[1].plot(fi.current_uA_cm2, fi.rate_brian2_hz, "s--", label="Brian2")
    axes[1].set_xlabel("I (uA/cm2)"); axes[1].set_ylabel("Rate (Hz)"); axes[1].legend(frameon=False); axes[1].set_title("HH f-I")
    axes[2].plot(lif.current_nA, lif.rate_analytic_hz, "k-", label="closed form")
    axes[2].plot(lif.current_nA, lif.rate_ours_hz, "o", label="NeuroSTIAS")
    if "rate_brian2_hz" in lif:
        axes[2].plot(lif.current_nA, lif.rate_brian2_hz, "s", mfc="none", label="Brian2")
    axes[2].set_xlabel("I (nA)"); axes[2].legend(frameon=False); axes[2].set_title("LIF f-I")
    exp.figure(fig, "e09_simulators.png")
    exp.summary.update({"hh_trace": hh_trace, "hh_fi": fi.to_dict("records"), "lif": lif.to_dict("records"),
                        "brunel": bsum.to_dict("records")})
    exp.note("Simulations are synthetic by nature; this experiment validates numerical correctness, not biology.")
    exp.note("Brian2 2.7.1 runs in an isolated environment with NumPy < 2 (Brian2 2.9 is incompatible with NumPy 2.4).")
    exp.finish([("Hodgkin-Huxley trace (10 uA/cm2, 100 ms)", "\n".join(f"- {k}: {v}" for k, v in hh_trace.items())),
                ("Hodgkin-Huxley f-I", md_table(fi)), ("LIF f-I", md_table(lif)), ("Brunel network (5 seeds, fixed in-degree in both simulators)", md_table(bsum)),
                ("Brunel time-step convergence, g=4.5, eta=0.9 (3 seeds)", md_table(conv_s))])
    print(hh_trace); print(fi); print(lif); print(bsum)


if __name__ == "__main__":
    main()
