"""E08 - Electrophysiology and EEG pipelines (validation experiment, real data).

Part A (DANDI 000006, mouse ALM units): NeuroSTIAS unit metrics (firing rate, ISI CV, LV,
Fano factor) vs Elephant's implementations on the same spike trains (agreement: max absolute
and relative difference, Pearson r). Plus a practical analysis: fraction of units whose
delay-epoch firing rates (pole retraction to go cue, good trials) differ between lick-left
and lick-right trials (Mann-Whitney,
BH-FDR across units) - descriptive.

Part B (EEGBCI, imagined hands vs feet, subjects 1-20): cross-validated decoding with
(1) the MNE reference pipeline (CSP + LDA, as in the MNE example) via NeuroSTIAS, and
(2) a simple NeuroSTIAS band-power (8-30 Hz, log power per channel) + logistic regression.
Chance level reported per subject.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neurostias import data  # noqa: E402
from neurostias.core.stats import bh  # noqa: E402
from neurostias.ephys.lfp_eeg import motor_imagery_csp_lda  # noqa: E402
from neurostias.ephys.spikes import unit_metrics  # noqa: E402
from neurostias.experiment import Experiment, md_table  # noqa: E402


def part_a(exp):
    import quantities as pq
    from elephant import statistics as es
    from neo import SpikeTrain
    from pynwb import NWBHDF5IO
    from scipy.stats import mannwhitneyu, pearsonr
    root = data.dataset_dir("dandi_000006")
    rows, sel_rows = [], []
    for f in sorted(root.rglob("*.nwb")):
        exp.prov.add_input(f, f.name, hash_file=True)
        with NWBHDF5IO(str(f), "r", load_namespaces=True) as io:
            nwb = io.read()
            if nwb.units is None or len(nwb.units) == 0:
                continue
            units = nwb.units.to_dataframe()
            trials = nwb.trials.to_dataframe() if nwb.trials is not None else None
            t_all = np.concatenate([np.asarray(s) for s in units["spike_times"]]) if len(units) else np.array([])
            t0, t1 = float(t_all.min()), float(t_all.max())
            for uid, row in units.iterrows():
                st = np.sort(np.asarray(row["spike_times"], float))
                if len(st) < 20:
                    continue
                ours = unit_metrics(st, t0, t1, fano_window=1.0)
                train = SpikeTrain(st * pq.s, t_start=t0 * pq.s, t_stop=(t1 + 1e-6) * pq.s)
                isi = es.isi(train)
                el = {"firing_rate_hz": float(es.mean_firing_rate(train).rescale(pq.Hz).magnitude),
                      "isi_cv": float(es.cv(isi.magnitude)), "lv": float(es.lv(isi.magnitude))}
                edges = np.arange(t0, t1 + 1.0, 1.0)
                counts = np.histogram(st, edges)[0]
                el["fano_factor"] = float(es.fanofactor([SpikeTrain(st[(st >= a) & (st < b)] * pq.s, t_start=a * pq.s, t_stop=b * pq.s)
                                                         for a, b in zip(edges[:-1], edges[1:])]))
                rows.append({"file": f.name, "unit": uid, **{f"ours_{k}": ours[k] for k in el}, **{f"elephant_{k}": v for k, v in el.items()}})
            if trials is not None and {"type", "pole_out_time", "cue_start_time", "start_time"} <= set(trials.columns):
                tr = trials[(trials.get("is_good", 1) == 1) & trials["type"].astype(str).str.startswith("lick")].copy()
                # delay epoch = pole retraction to go cue; times in this dandiset are relative to trial start
                d0 = (tr["start_time"] + tr["pole_out_time"]).to_numpy(float)
                d1 = (tr["start_time"] + tr["cue_start_time"]).to_numpy(float)
                ok = np.isfinite(d0) & np.isfinite(d1) & (d1 > d0)
                labs = tr["type"].astype(str).to_numpy()[ok]
                d0, d1 = d0[ok], d1[ok]
                u = np.unique(labs)
                if len(u) == 2 and min((labs == u[0]).sum(), (labs == u[1]).sum()) >= 5:
                    for uid, row in units.iterrows():
                        st = np.sort(np.asarray(row["spike_times"], float))
                        if len(st) < 20:
                            continue
                        rate = (np.searchsorted(st, d1) - np.searchsorted(st, d0)) / (d1 - d0)
                        p = mannwhitneyu(rate[labs == u[0]], rate[labs == u[1]]).pvalue
                        sel_rows.append({"file": f.name, "unit": uid, "n_trials": int(len(labs)), "pval": float(p)})
    df = pd.DataFrame(rows)
    agree = []
    for k in ("firing_rate_hz", "isi_cv", "lv", "fano_factor"):
        a, b = df[f"ours_{k}"].to_numpy(), df[f"elephant_{k}"].to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        agree.append({"metric": k, "n_units": int(m.sum()), "max_abs_diff": float(np.max(np.abs(a[m] - b[m]))),
                      "max_rel_diff": float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-12))),
                      "pearson_r": float(pearsonr(a[m], b[m])[0])})
    sel = pd.DataFrame(sel_rows)
    sel_summary = {}
    if len(sel):
        sel["padj"] = bh(sel["pval"])
        sel_summary = {"n_units_tested": int(len(sel)), "frac_selective_fdr05": float((sel["padj"] < 0.05).mean())}
    return df, pd.DataFrame(agree), sel, sel_summary


def part_b(exp):
    import mne
    from mne.datasets import eegbci
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import ShuffleSplit, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    root = data.dataset_dir("ephys_imaging") / "eegbci"
    rows = []
    for s in range(1, 21):
        try:
            files = eegbci.load_data(s, [6, 10, 14], path=str(root), update_path=False, verbose="ERROR")
        except Exception as exc:  # noqa: BLE001
            exp.note(f"EEGBCI subject {s} unavailable: {exc}")
            continue
        raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose="ERROR") for f in files])
        eegbci.standardize(raw)
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
        raw.annotations.rename(dict(T1="hands", T2="feet"))
        events, ev_id = mne.events_from_annotations(raw, event_id=dict(hands=2, feet=3), verbose="ERROR")
        r = motor_imagery_csp_lda(raw, events, ev_id)
        # band-power + logistic regression on the same epochs
        rf = raw.copy().filter(8.0, 30.0, fir_design="firwin", verbose="ERROR")
        ep = mne.Epochs(rf, events, ev_id, 1.0, 2.0, baseline=None, preload=True, picks="eeg", verbose="ERROR")
        X = np.log(np.var(ep.get_data(copy=False), axis=2))
        y = ep.events[:, -1]
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        bp = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=cv)
        rows.append({"subject": s, "csp_lda_accuracy": r["accuracy_mean"], "bandpower_logreg_accuracy": float(bp.mean()),
                     "chance": r["chance"], "n_epochs": r["n_epochs"]})
        print("eeg subject", s, flush=True)
    return pd.DataFrame(rows)


def main():
    exp = Experiment("E08", "Spike-train and EEG analysis validation", "Do NeuroSTIAS spike-train metrics agree with Elephant, "
                     "and do its EEG decoding pipelines reach reference accuracy on real motor-imagery data?",
                     params={"fano_window_s": 1.0, "eeg_band_hz": [8, 30], "cv": "ShuffleSplit(10, 0.2)"})
    units, agree, sel, sel_summary = part_a(exp)
    exp.table(units, "e08a_unit_metrics_ours_vs_elephant.csv")
    exp.table(agree, "e08a_agreement.csv")
    if len(sel):
        exp.table(sel, "e08a_delay_selectivity.csv")
    eeg = part_b(exp)
    exp.table(eeg, "e08b_eegbci_decoding.csv")
    eeg_sum = eeg[["csp_lda_accuracy", "bandpower_logreg_accuracy", "chance"]].agg(["mean", "std"]).T.reset_index()
    exp.table(eeg_sum, "e08b_eegbci_summary.csv")
    exp.summary.update({"agreement": agree.to_dict("records"), "selectivity": sel_summary, "eeg": eeg_sum.to_dict("records"),
                        "n_units": int(len(units))})
    exp.note("EEGBCI subject 1 CSP+LDA accuracy can be compared with the MNE documentation example for the same "
             "subject and runs.")
    exp.note("Validation experiment: NeuroSTIAS wraps or re-implements established methods; agreement, not superiority, "
             "is the claim.")
    exp.finish([("Part A - unit metrics: NeuroSTIAS vs Elephant", md_table(agree)),
                ("Part A - delay-epoch selectivity (descriptive)", "\n".join(f"- {k}: {v}" for k, v in sel_summary.items()) or "trial fields not available"),
                ("Part B - EEGBCI decoding (mean, SD over subjects)", md_table(eeg_sum)),
                ("Part B - per subject", md_table(eeg))])
    print(agree); print(sel_summary); print(eeg_sum)


if __name__ == "__main__":
    main()
