"""E07 - Brain-organoid maturation and regional identity (real data, HNOCA time-course subset).

Ground truth: organoid age in days (HNOCA metadata) and HNOCA's harmonised annotations
(annot_level_2 cell states, annot_region_rev2 regions), which were derived by the atlas
authors by reference mapping - an independent source from NeuroSTIAS panels.

Part A - maturation vs age. Per bio_sample (independent organoid sample) the median of:
  neurostias_maturation   mature-neuron minus radial-glia/NPC enrichment effect
  stemness_v441           3D-STIAS V4.4.1 pluripotency composite (expected to fall with age)
  genes_detected          CytoTRACE principle (transcriptional diversity falls with differentiation)
Spearman correlation with age across samples (n = number of bio_samples), 95% bootstrap CI.

Part B - cell-state annotation vs annot_level_2 (mapped to NeuroSTIAS neurodevelopmental
stages) and Part C - regional identity of neural cells vs annot_region_rev2 (mapped to the
developing_regions panels). Methods: NeuroSTIAS test (Unresolved allowed) vs plain argmax.
Metrics: coverage, accuracy among resolved cells, macro-F1.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neurostias import data  # noqa: E402
from neurostias.atlas import markers  # noqa: E402
from neurostias.atlas.annotate import call_labels, score_panels  # noqa: E402
from neurostias.experiment import Experiment, md_table  # noqa: E402
from neurostias.legacy.development import neurodevelopment_stages, stemness_v441  # noqa: E402


def stage_map(lbl: str) -> str:
    l = str(lbl).lower()
    if "ipc" in l or "intermediate" in l:
        return "Intermediate progenitor"
    if "npc" in l or "radial" in l or "progenitor" in l:
        return "Radial glia / NPC"
    if "neuron" in l:
        return "Neuron"
    if "glio" in l or "opc" in l or "astro" in l or "oligo" in l:
        return "Glial progenitor"
    return "other"


def region_map(lbl: str) -> str:
    l = str(lbl).lower()
    table = [("dorsal telen", "Dorsal telencephalon"), ("ventral telen", "Ventral telencephalon"),
             ("hypothal", "Hypothalamus"), ("thalam", "Thalamus / diencephalon"), ("dienceph", "Thalamus / diencephalon"),
             ("midbrain", "Midbrain"), ("hindbrain", "Hindbrain"), ("cerebell", "Cerebellum"), ("retina", "Retina"),
             ("choroid", "Choroid plexus"), ("crest", "Neural crest / mesenchyme"), ("mesench", "Neural crest / mesenchyme")]
    for k, v in table:
        if k in l:
            return v
    return "other"


def boot_spearman(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    r0 = spearmanr(x, y)[0]
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) > 2:
            bs.append(spearmanr(x[i], y[i])[0])
    return float(r0), float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975)), float(spearmanr(x, y)[1])


def main():
    exp = Experiment("E07", "Brain-organoid maturation and regional identity", "Do NeuroSTIAS neurodevelopmental and regional "
                     "panels track organoid age and recover atlas-level cell states and regions better than V4.4.1 stemness "
                     "and simple alternatives?", params={"alpha": 0.05, "n_null": 1000})
    p = data.path("organoid_hnoca")
    exp.prov.add_input(p, "hnoca_timecourse_subset")
    a = sc.read_h5ad(p)
    exp.summary["selection"] = a.uns.get("selection")
    a.X = a.layers["counts"].copy()
    a.obs["genes_detected"] = np.asarray((a.X > 0).sum(1)).ravel()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.layers["log_normalized"] = a.X.copy()
    neurodevelopment_stages(a, species="human", seed=0)
    stemness_v441(a)
    a.obs["age"] = pd.to_numeric(a.obs["organoid_age_days"], errors="coerce")
    # ---- Part A
    samp = a.obs.groupby("bio_sample", observed=True).agg(age=("age", "first"), maturation=("maturation_index", "median"),
                                                           stemness_v441=("stemness_composite", "median") if "stemness_composite" in a.obs else ("age", "first"),
                                                           genes_detected=("genes_detected", "median"), n_cells=("age", "size")).dropna(subset=["age"])
    exp.table(samp.reset_index(), "e07a_sample_level.csv")
    rows = []
    for col in ("maturation", "stemness_v441", "genes_detected"):
        if col == "stemness_v441" and "stemness_composite" not in a.obs:
            continue
        r, lo, hi, pv = boot_spearman(samp["age"].to_numpy(), samp[col].to_numpy())
        rows.append({"score": col, "spearman_vs_age": r, "ci_low": lo, "ci_high": hi, "pval": pv, "n_samples": len(samp),
                     "expected_sign": "+" if col == "maturation" else "-"})
    part_a = pd.DataFrame(rows)
    exp.table(part_a, "e07a_maturation_vs_age.csv")
    # Post-hoc (added after seeing Part A, reported separately): sample neuron fractions.
    # atlas_neuron_fraction uses HNOCA's own labels and is a reference ceiling, not a NeuroSTIAS result.
    a.obs["_ns_neuron"] = a.obs["neurodev_stage"].astype(str).isin(["Immature neuron", "Mature neuron"])
    a.obs["_atlas_neuron"] = a.obs["annot_level_1"].astype(str) == "Neuron"
    frac = a.obs.groupby("bio_sample", observed=True).agg(age=("age", "first"), neurostias_neuron_fraction=("_ns_neuron", "mean"),
                                                           atlas_neuron_fraction=("_atlas_neuron", "mean")).dropna(subset=["age"])
    ph = []
    for col in ("neurostias_neuron_fraction", "atlas_neuron_fraction"):
        r, lo, hi, pv = boot_spearman(frac["age"].to_numpy(), frac[col].to_numpy())
        ph.append({"score": col, "spearman_vs_age": r, "ci_low": lo, "ci_high": hi, "pval": pv, "n_samples": len(frac),
                   "expected_sign": "+", "analysis": "post hoc" if col.startswith("neurostias") else "reference (atlas labels)"})
    part_a_ph = pd.DataFrame(ph)
    exp.table(part_a_ph, "e07a_posthoc_neuron_fraction.csv")
    # Selection check: are the first <=600 cells per sample representative of the whole sample?
    full_obs = p.parent / "obs_cache.pkl"
    sel_check = None
    if full_obs.exists():
        full = pd.read_pickle(full_obs)
        full = full[full.publication.isin(a.obs["publication"].astype(str).unique())]
        cf = pd.crosstab(full.bio_sample, full.annot_level_1, normalize="index")
        cs = pd.crosstab(a.obs.bio_sample.astype(str), a.obs.annot_level_1.astype(str), normalize="index")
        cs = cs.reindex(columns=cf.columns, fill_value=0).reindex(cf.index).fillna(0)
        tv = 0.5 * (cf - cs).abs().sum(1)  # total-variation distance of class composition, per sample
        age_f = full.groupby("bio_sample").organoid_age_days.first().astype(float)
        sel_check = pd.DataFrame([{
            "tv_distance_median": float(tv.median()), "tv_distance_max": float(tv.max()), "n_samples": int(len(tv)),
            "spearman_age_neuron_fraction_full": float(spearmanr(age_f.loc[cf.index], cf.get("Neuron", 0))[0]),
            "spearman_age_neuron_fraction_subset": float(spearmanr(age_f.loc[cs.index], cs.get("Neuron", 0))[0])}])
        exp.table(sel_check, "e07_selection_check.csv")
    # ---- Part B: stages
    truth_b = a.obs["annot_level_2"].astype(str).map(stage_map).to_numpy()
    ours_b = a.obs["neurodev_stage"].astype(str).replace({"Immature neuron": "Neuron", "Mature neuron": "Neuron",
                                                           "Cycling progenitor": "Radial glia / NPC"}).to_numpy()
    eff, pv, _ = score_panels(a, markers.panels("neurodevelopment", "human"), n_null=1000, seed=0)
    am = pd.Series(eff.columns.to_numpy()[np.argmax(eff.to_numpy(), 1)]).replace(
        {"Immature neuron": "Neuron", "Mature neuron": "Neuron", "Cycling progenitor": "Radial glia / NPC"}).to_numpy()
    classes_b = ["Radial glia / NPC", "Intermediate progenitor", "Neuron", "Glial progenitor"]
    def met(t, pr, classes):
        ev = np.isin(t, classes)
        res = pr != "Unresolved"
        return {"coverage": float(np.mean(res[ev])),
                "accuracy_resolved": float(np.mean(pr[ev & res] == t[ev & res])) if (ev & res).any() else np.nan,
                "accuracy_overall": float(np.mean(pr[ev] == t[ev])),
                "macro_f1": float(f1_score(t[ev], pr[ev], labels=classes, average="macro", zero_division=0)), "n_cells": int(ev.sum())}
    part_b = pd.DataFrame([{"method": "neurostias_test", **met(truth_b, ours_b, classes_b)},
                           {"method": "argmax", **met(truth_b, am, classes_b)}])
    exp.table(part_b, "e07b_stage_annotation.csv")
    exp.summary["annot_level_2_values"] = a.obs["annot_level_2"].astype(str).value_counts().to_dict()
    # ---- Part C: regions (neural cells only)
    truth_c = a.obs["annot_region_rev2"].astype(str).map(region_map).to_numpy()
    e2, p2, cov2 = score_panels(a, markers.panels("developing_regions", "human"), n_null=1000, seed=0)
    lab2, _ = call_labels(e2, p2, 0.05)
    am2 = e2.columns.to_numpy()[np.argmax(e2.to_numpy(), 1)]
    classes_c = sorted(set(truth_c) - {"other"})
    part_c = pd.DataFrame([{"method": "neurostias_test", **met(truth_c, lab2.astype(object), classes_c)},
                           {"method": "argmax", **met(truth_c, am2.astype(object), classes_c)}])
    exp.table(part_c, "e07c_region_annotation.csv")
    exp.table(cov2, "e07c_region_panel_coverage.csv")
    exp.summary["annot_region_values"] = a.obs["annot_region_rev2"].astype(str).value_counts().to_dict()
    exp.summary.update({"part_a": part_a.to_dict("records"), "part_b": part_b.to_dict("records"), "part_c": part_c.to_dict("records"),
                        "n_cells": int(a.n_obs), "n_samples": int(len(samp)), "ages": sorted(samp["age"].unique().tolist())})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, col in zip(axes, ("maturation", "stemness_v441", "genes_detected")):
        if col in samp:
            ax.scatter(samp["age"], samp[col], s=14)
            ax.set_xlabel("Organoid age (days)"); ax.set_ylabel(col); ax.set_xscale("log")
    exp.figure(fig, "e07a_scores_vs_age.png")
    exp.note("Sample-level statistics (one value per bio_sample) are primary; cell-level correlations would treat cells "
             "of one organoid as independent.")
    exp.note("Post-hoc analysis: after Part A showed no age trend for the pre-specified maturation index, sample neuron "
             "fractions were added. The NeuroSTIAS neuron fraction is post hoc and should be read as exploratory; the "
             "atlas neuron fraction uses HNOCA's own labels and is a reference ceiling.")
    exp.note("Selection check: the subset takes the first <=600 cells of each bio_sample in atlas order; class composition "
             "is compared with the full sample (total-variation distance) and the age-neuron fraction trend is recomputed "
             "on the full samples.")
    exp.note("Paulsen et al. 2022 organoids include several cell lines and autism-gene mutants alongside controls, so age "
             "trends are pooled over genotypes.")
    exp.note("HNOCA annotations come from reference mapping by the atlas authors; agreement measures consistency with "
             "the atlas, not absolute truth.")
    exp.summary["part_a_posthoc"] = part_a_ph.to_dict("records")
    if sel_check is not None:
        exp.summary["selection_check"] = sel_check.to_dict("records")
    exp.finish([("Part A - maturation scores vs organoid age (sample level)", md_table(part_a)),
                ("Part A (post hoc) - neuron fraction vs organoid age", md_table(part_a_ph)),
                ("Selection check - subset vs full sample composition", md_table(sel_check) if sel_check is not None else "not available"),
                ("Part B - cell-state annotation vs HNOCA annot_level_2", md_table(part_b)),
                ("Part C - regional identity vs HNOCA annot_region_rev2", md_table(part_c))])
    print(part_a); print(part_b); print(part_c)


if __name__ == "__main__":
    main()
