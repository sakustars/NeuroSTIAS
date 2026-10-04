"""E10 - Neuron morphology (validation + practical classification, real data).

NeuroMorpho.org Allen Cell Types reconstructions (balanced principal cells vs interneurons).
Part A: NeuroSTIAS morphometrics vs NeuroM (total neurite length, number of bifurcations,
number of tips/terminations, number of stems, max radial distance) - Pearson r and median
relative difference per metric.
Part B: principal-vs-interneuron classification from morphometrics (random forest, stratified
10-fold CV, 5 repeats) using NeuroSTIAS features vs NeuroM features; chance = majority class.
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
from neurostias.experiment import Experiment, md_table  # noqa: E402
from neurostias.morphology.swc import morphometrics, read_swc, sholl  # noqa: E402


def neurom_feats(path):
    import neurom as nm
    n = nm.load_morphology(str(path))
    return {"total_length": float(np.sum(nm.get("total_length_per_neurite", n))),
            "n_branch_points": int(nm.get("number_of_bifurcations", n)),
            "n_tips": int(nm.get("number_of_leaves", n)),
            "n_stems": int(nm.get("number_of_neurites", n)),
            "max_radial_distance": float(nm.get("max_radial_distance", n))}


def main():
    exp = Experiment("E10", "Neuron morphometrics and cell-class classification", "Do NeuroSTIAS morphometrics agree with "
                     "NeuroM, and can they separate principal cells from interneurons?", params={"cv": "10-fold x 5"})
    root = data.dataset_dir("neuromorpho")
    meta = pd.read_csv(root / "selected_metadata.csv")
    exp.prov.add_input(root / "selected_metadata.csv", "neuromorpho_metadata")
    rows, fails = [], 0
    for _, r in meta.iterrows():
        p = root / "swc" / f"{r.neuron_name}.CNG.swc"
        if not p.exists():
            continue
        try:
            ours = morphometrics(read_swc(p))
            nmf = neurom_feats(p)
            sh = sholl(read_swc(p), step=20.0)
            rows.append({"neuron": r.neuron_name, "cls": r.cls, **{f"ours_{k}": ours[k] for k in nmf},
                         **{f"neurom_{k}": v for k, v in nmf.items()},
                         "ours_sholl_max": int(sh["intersections"].max()),
                         "ours_sholl_radius_of_max": float(sh.loc[sh["intersections"].idxmax(), "radius"]),
                         "ours_path_length_max": ours["path_length_max"],
                         "ours_axon_length": ours["axon_length"], "ours_dendrite_length": ours["basal_dendrite_length"] + ours["apical_dendrite_length"],
                         "ours_apical_length": ours["apical_dendrite_length"]})
        except Exception as exc:  # noqa: BLE001
            fails += 1
            exp.note(f"failed {r.neuron_name}: {str(exc)[:120]}")
    df = pd.DataFrame(rows)
    exp.table(df, "e10_morphometrics.csv")
    from scipy.stats import pearsonr
    agree = []
    for k in ("total_length", "n_branch_points", "n_tips", "n_stems", "max_radial_distance"):
        a, b = df[f"ours_{k}"].astype(float), df[f"neurom_{k}"].astype(float)
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-9)
        agree.append({"metric": k, "n": len(df), "pearson_r": float(pearsonr(a, b)[0]), "median_rel_diff": float(rel.median()),
                      "frac_identical": float(np.mean(np.isclose(a, b, rtol=1e-6)))})
    agree = pd.DataFrame(agree)
    exp.table(agree, "e10_agreement_vs_neurom.csv")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    y = (df["cls"] == "principal").astype(int).to_numpy()
    feats = {"neurostias": [c for c in df.columns if c.startswith("ours_")],
             "neurom": [c for c in df.columns if c.startswith("neurom_")]}
    cls_rows = []
    for name, cols in feats.items():
        X = df[cols].to_numpy(float)
        sc = cross_val_score(RandomForestClassifier(300, random_state=0), X, y,
                             cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0), scoring="balanced_accuracy")
        cls_rows.append({"features": name, "n_features": len(cols), "balanced_accuracy_mean": float(sc.mean()),
                         "balanced_accuracy_sd": float(sc.std()), "chance": 0.5})
    cls = pd.DataFrame(cls_rows)
    exp.table(cls, "e10_classification.csv")
    exp.summary.update({"n_neurons": int(len(df)), "class_counts": df["cls"].value_counts().to_dict(), "agreement": agree.to_dict("records"),
                        "classification": cls.to_dict("records"), "n_failed": fails})
    exp.note("The classification comparison is not feature-matched: the NeuroSTIAS set has 11 features (adds Sholl "
             "peak, path length and axon/dendrite lengths) versus 5 NeuroM features; the accuracy gap reflects feature "
             "choice, not a better implementation. Neurons that NeuroM/MorphIO rejects as malformed are excluded from "
             "both arms (listed above).")
    exp.note("NeuroM's number_of_leaves counts terminations including the axon; differences in soma handling can make "
             "max radial distance differ slightly (NeuroSTIAS measures from the soma centroid).")
    exp.finish([("Agreement with NeuroM", md_table(agree)), ("Principal vs interneuron classification", md_table(cls))])
    print(agree); print(cls)


if __name__ == "__main__":
    main()
