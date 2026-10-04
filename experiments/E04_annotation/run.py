"""E04 - Brain cell-type annotation against independent ground truth.

Part A (MERFISH hypothalamus; author cell classes, Moffitt et al. 2018). Author classes are
mapped to NeuroSTIAS classes (Excitatory/Inhibitory neuron, Astrocyte, Oligodendrocyte lineage
[OD Mature + OD Immature, since immature ODs/OPCs are not separable in the author labels],
Microglia, Endothelial, Pericyte, Ependymal). Methods:
  neurostias            hierarchical panels + expression-matched control-set test (Unresolved allowed)
  neurostias_argmax     same panels, plain argmax of mean z-scores (ablation: no statistical test)
  v441_generic          3D-STIAS V4.4.1 generic panels (immune/skin/brain), argmax of mean z-scores
  celltypist            Mouse_Whole_Brain model (isolated environment), labels mapped to classes
Metrics: coverage (fraction not Unresolved), accuracy among resolved cells, overall accuracy
(Unresolved counted wrong), macro-F1 over the evaluable classes.

Part B (DLPFC; manual layers). Canonical layer panels (literature, not derived from DLPFC) are
scored per spot; spots are labelled with the best significant layer. Compared with a
plain argmax of the same panels. Metric: accuracy per section vs manual layers (L1-L6, WM).

Part C (Human Brain 3D, no ground truth): V4.4.1 labels vs NeuroSTIAS classes and panel
coverage - descriptive only.
"""
import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neurostias import data  # noqa: E402
from neurostias.atlas import markers  # noqa: E402
from neurostias.atlas.annotate import annotate, call_labels, score_panels  # noqa: E402
from neurostias.experiment import Experiment, md_table, run_baseline_script  # noqa: E402

AUTHOR_MAP = {"Excitatory": "Excitatory neuron", "Inhibitory": "Inhibitory neuron", "Astrocyte": "Astrocyte",
              "OD Mature": "Oligodendrocyte lineage", "OD Immature": "Oligodendrocyte lineage", "Microglia": "Microglia",
              "Endothelial": "Endothelial", "Pericytes": "Pericyte", "Ependymal": "Ependymal"}
OURS_MAP = {"Oligodendrocyte": "Oligodendrocyte lineage", "OPC": "Oligodendrocyte lineage"}
V441_PANELS = {  # 3D-STIAS V4.4.1 STANDARD_MARKER_PANELS (verbatim)
    "Neuron": ("RBFOX3", "SNAP25", "SYT1", "TUBB3", "MAP2", "NEFL", "SLC17A7", "GAD1"),
    "Oligodendrocyte": ("MBP", "PLP1", "MOG", "MAG", "MOBP", "CNP", "CLDN11", "SOX10"),
    "Oligodendrocyte precursor cell": ("PDGFRA", "CSPG4", "OLIG1", "OLIG2", "SOX10", "VCAN"),
    "Astrocyte": ("GFAP", "AQP4", "ALDH1L1", "SLC1A2", "SLC1A3", "GLUL", "SOX9"),
    "Microglia": ("C1QA", "C1QB", "C1QC", "TYROBP", "AIF1", "LST1", "CTSS", "FCER1G"),
    "B cell": ("MS4A1", "CD79A", "CD79B", "CD37", "CD74", "CD22", "CD19", "CD83"),
    "Plasma cell": ("MZB1", "JCHAIN", "SDC1", "IGHG1", "IGKC", "CD79A"),
    "Plasmacytoid dendritic": ("GZMB", "LILRA4", "CLEC4C", "IL3RA", "IRF7", "TCF4", "JCHAIN"),
    "CD4 T cell": ("CD3D", "CD3E", "TRAC", "IL7R", "CCR7", "LTB", "MAL"),
    "Cytotoxic T/NK": ("NKG7", "GNLY", "KLRD1", "CCL5", "GZMB", "GZMH", "CTSW"),
    "Classical monocyte": ("LYZ", "S100A8", "S100A9", "S100A12", "FCN1", "CD14", "VCAN"),
    "Non-classical monocyte": ("LST1", "FCGR3A", "MS4A7", "LILRB1", "IFITM3", "SERPINA1", "CTSS"),
    "Macrophage": ("C1QA", "C1QB", "C1QC", "FOLR2", "APOE", "F13A1", "SELENOP"),
    "Dendritic/APC": ("HLA-DRA", "HLA-DPA1", "HLA-DPB1", "CD74", "FCER1A", "CD1C", "CLEC10A"),
    "Mast cell": ("TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2", "HPGDS", "CTSG"),
    "Endothelial": ("PECAM1", "VWF", "EMCN", "KDR", "EGFL7", "RAMP2", "CLDN5"),
    "Lymphatic endothelial": ("CCL21", "PDPN", "PROX1", "MMRN1", "LYVE1", "FLT4", "CLDN5"),
    "Fibroblast": ("COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "COL6A1", "COL6A2"),
    "Inflammatory fibroblast": ("COL1A1", "COL3A1", "LUM", "CXCL12", "SOD2", "TNFAIP6", "NNMT"),
    "Pericyte": ("RGS5", "CSPG4", "MCAM", "PDGFRB", "NOTCH3"),
    "Vascular smooth muscle cell": ("MYL9", "MYH11", "TAGLN", "ACTA2", "DES"),
    "Myofibroblast": ("ACTA2", "TAGLN", "DES", "TGFBI", "FN1", "COL1A1", "COL3A1"),
    "Basal keratinocyte": ("KRT5", "KRT14", "KRT15", "COL17A1", "KRT17", "TP63", "S100A2"),
    "Differentiated keratinocyte": ("KRT1", "KRT10", "KRTDAP", "DMKN", "DSG1", "SBSN", "CALML5"),
    "Glandular epithelial": ("KRT8", "KRT18", "KRT19", "KRT7", "EPCAM", "MUC1", "KRT4"),
    "Melanocyte": ("MLANA", "PMEL", "TYR", "TYRP1", "DCT", "SOX10", "MITF"),
    "Schwann cell": ("S100B", "SOX10", "PLP1", "PMP22", "S100A1", "MPZ", "NGFR"),
    "Osteoclast-like myeloid": ("ACP5", "CTSK", "MMP9", "SPP1", "TCIRG1", "SIGLEC15", "ATP6V0D2"),
}
V441_TO_CLASS = {"Neuron": "Neuron (unspecified)", "Oligodendrocyte": "Oligodendrocyte lineage",
                 "Oligodendrocyte precursor cell": "Oligodendrocyte lineage", "Astrocyte": "Astrocyte",
                 "Microglia": "Microglia", "Endothelial": "Endothelial", "Pericyte": "Pericyte"}


def argmax_zscore(adata, panels, min_genes=2):
    """V4.4.1-style scoring: mean z-score of panel genes (case-insensitive), label = argmax."""
    X = adata.layers["log_normalized"] if "log_normalized" in adata.layers else adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    up = {g.upper(): i for i, g in enumerate(adata.var_names.astype(str))}
    usable = {k: [up[g.upper()] for g in v if g.upper() in up] for k, v in panels.items()}
    usable = {k: v for k, v in usable.items() if len(v) >= min_genes}
    cols = sorted({i for v in usable.values() for i in v})
    Z = X[:, cols]
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 1e-12, Z.std(0), 1)
    pos = {c: j for j, c in enumerate(cols)}
    S = pd.DataFrame({k: Z[:, [pos[i] for i in v]].mean(1) for k, v in usable.items()})
    return S.columns.to_numpy()[np.argmax(S.to_numpy(), axis=1)], list(usable)


def metrics(truth, pred, classes):
    pred = np.asarray(pred, dtype=object)
    resolved = ~np.isin(pred, ["Unresolved"])
    ev = np.isin(truth, classes)
    cov = float(np.mean(resolved[ev]))
    acc_res = float(np.mean(pred[ev & resolved] == truth[ev & resolved])) if (ev & resolved).any() else float("nan")
    acc_all = float(np.mean(pred[ev] == truth[ev]))
    f1 = f1_score(truth[ev], pred[ev], labels=classes, average="macro", zero_division=0)
    per = {c: float(np.mean(pred[ev & (truth == c)] == c)) for c in classes}
    return {"coverage": cov, "accuracy_resolved": acc_res, "accuracy_overall": acc_all, "macro_f1": float(f1),
            **{f"recall_{c}": v for c, v in per.items()}}


def part_a(exp):
    p = data.path("merfish_hypothalamus")
    exp.prov.add_input(p, "merfish_moffitt2018")
    a = sc.read_h5ad(p)
    a.obs["author_class"] = a.obs["Cell_class"].astype(str).str.replace(r" \d+$", "", regex=True)
    a = a[a.obs["author_class"].isin(AUTHOR_MAP)].copy()
    truth = a.obs["author_class"].map(AUTHOR_MAP).to_numpy()
    # MERFISH values are volume-normalised counts: log1p after scaling to the median library size
    import scipy.sparse as sp
    X = a.X.tocsr()
    lib = np.asarray(X.sum(1)).ravel()
    a.layers["log_normalized"] = sp.csr_matrix(np.log1p(X.multiply(np.median(lib) / np.maximum(lib, 1e-9)[:, None]).toarray()))
    classes = sorted(set(AUTHOR_MAP.values()))
    rows, panel_cov = [], None
    res = annotate(a, species="mouse", seed=0)
    panel_cov = res["coverage"]
    ours = a.obs["neuro_class"].astype(str).replace(OURS_MAP).to_numpy()
    rows.append({"method": "neurostias", **metrics(truth, ours, classes)})
    # sensitivity: allow 2-gene panels (the default requires 3 detected genes and abstains otherwise;
    # the 161-gene MERFISH panel contains at most 2 genes of each NeuroSTIAS class panel)
    annotate(a, species="mouse", seed=0, min_genes=2)
    ours2 = a.obs["neuro_class"].astype(str).replace(OURS_MAP).to_numpy()
    rows.append({"method": "neurostias_min_genes2", **metrics(truth, ours2, classes)})
    am, used = argmax_zscore(a, markers.panels("classes", "mouse"))
    rows.append({"method": "neurostias_argmax", **metrics(truth, pd.Series(am).replace(OURS_MAP).to_numpy(), classes)})
    v, used_v = argmax_zscore(a, V441_PANELS)
    v = pd.Series(v).map(lambda x: V441_TO_CLASS.get(x, f"non-brain: {x}")).to_numpy()
    rows.append({"method": "v441_generic", **metrics(truth, v, classes)})
    exp.summary["v441_label_distribution_merfish"] = pd.Series(v).value_counts().to_dict()
    # CellTypist
    with tempfile.TemporaryDirectory() as td:
        ip, op = Path(td) / "x.h5ad", Path(td) / "o.csv"
        b = sc.AnnData(X=a.layers["log_normalized"], obs=a.obs[[]].copy())
        b.var_names = a.var_names
        b.write_h5ad(ip)
        ok, log = run_baseline_script(ROOT / "benchmarks/baselines/celltypist_annotate.py",
                                      ["--input", str(ip), "--model", "Mouse_Whole_Brain.pkl", "--out", str(op)], timeout=3600)
        if ok:
            ct = pd.read_csv(op)["label"].astype(str)
            exp.summary["celltypist_raw_labels_top"] = ct.value_counts().head(25).to_dict()
            exp.summary["celltypist_log"] = log[-300:]
            ct_map = map_celltypist(ct)
            rows.append({"method": "celltypist_Mouse_Whole_Brain", **metrics(truth, ct_map, classes)})
            exp.baseline("celltypist", "ran", log[-300:])
        else:
            exp.baseline("celltypist", "failed", log)
    return pd.DataFrame(rows), panel_cov


def map_celltypist(labels: pd.Series) -> np.ndarray:
    """Map Allen whole-brain taxonomy class names to NeuroSTIAS classes by keyword (documented rule)."""
    def f(s):
        l = s.lower()
        if "glut" in l:
            return "Excitatory neuron"
        if "gaba" in l or "inh" in l:
            return "Inhibitory neuron"
        if "astro" in l:
            return "Astrocyte"
        if "oligo" in l or "opc" in l or "olig" in l:
            return "Oligodendrocyte lineage"
        if "microglia" in l:
            return "Microglia"
        if "endo" in l:
            return "Endothelial"
        if "peri" in l:
            return "Pericyte"
        if "epen" in l:
            return "Ependymal"
        return f"other: {s}"
    return labels.map(f).to_numpy()


def part_b(exp):
    rows = []
    for s in ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675", "151676"]:
        a = sc.read_h5ad(data.path("dlpfc", f"{s}.h5ad"))
        a.X = a.layers["counts"]
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a.layers["log_normalized"] = a.X.copy()
        truth = a.obs["layer"].astype(str).to_numpy()
        pnl = markers.panels("cortical_layers", "human")
        name_map = {"L1": "L1", "L2/3": "L2/3", "L4": "L4", "L5": "L5", "L6": "L6", "White matter": "WM"}
        truth_m = np.where(np.isin(truth, ["L2", "L3"]), "L2/3", truth)
        e, p, cov = score_panels(a, pnl, n_null=1000, seed=0)
        lab, _ = call_labels(e, p, 0.05)
        lab = pd.Series(lab).replace(name_map).to_numpy()
        am = pd.Series(e.columns.to_numpy()[np.argmax(e.to_numpy(), axis=1)]).replace(name_map).to_numpy()
        ev = truth != "NA"
        for meth, pred in (("neurostias_layer_test", lab), ("layer_panel_argmax", am)):
            resolved = pred != "Unresolved"
            rows.append({"section": s, "method": meth, "coverage": float(np.mean(resolved[ev])),
                         "accuracy_resolved": float(np.mean(pred[ev & resolved] == truth_m[ev & resolved])),
                         "accuracy_overall": float(np.mean(pred[ev] == truth_m[ev])),
                         "chance_majority_layer": float(pd.Series(truth_m[ev]).value_counts(normalize=True).iloc[0])})
    return pd.DataFrame(rows)


def part_c(exp):
    try:
        p = data.path("human_brain_3d")
    except (KeyError, FileNotFoundError):
        return {"status": "private dataset not available; Part C skipped"}
    exp.prov.add_input(p, "human_brain_3d")
    a = sc.read_h5ad(p)
    res = annotate(a, seed=0)
    return {"v441_labels": a.obs["cell_type"].astype(str).value_counts().to_dict(),
            "neurostias_classes": res["summary"]["class_counts"],
            "unusable_panels_in_this_gene_list": res["summary"]["unusable_class_panels"],
            "fraction_unresolved": res["summary"]["fraction_unresolved"]}


def main(quick):
    exp = Experiment("E04", "Brain cell-type annotation", "Do curated brain panels with a statistical test annotate brain "
                     "cell classes and cortical layers more accurately than the generic V4.4.1 panels, plain argmax, "
                     "and an established reference classifier (CellTypist)?", params={"alpha": 0.05, "n_null": 1000})
    a_res, cov = part_a(exp)
    exp.table(a_res, "e04a_merfish_metrics.csv")
    exp.table(cov, "e04a_merfish_panel_coverage.csv")
    b_res = part_b(exp)
    exp.table(b_res, "e04b_dlpfc_layers_per_section.csv")
    b_sum = b_res.groupby("method", as_index=False)[["coverage", "accuracy_resolved", "accuracy_overall",
                                                     "chance_majority_layer"]].mean()
    exp.table(b_sum, "e04b_dlpfc_layers_summary.csv")
    c = part_c(exp)
    # Part C uses a privately shared dataset (source/licence unconfirmed): kept in a separate private file,
    # excluded from the public release by benchmarks/make_public_release.py
    import json as _json
    (exp.res / "private_e04c_human_brain_3d.json").write_text(_json.dumps(c, indent=2, default=str))
    exp.summary.update({"part_a": a_res.to_dict("records"), "part_b": b_sum.to_dict("records")})
    exp.note("MERFISH measures a 161-gene targeted panel, so several markers are absent; panel coverage is reported in "
             "e04a_merfish_panel_coverage.csv. CellTypist models are built on whole transcriptomes and are evaluated on "
             "the overlapping genes only.")
    exp.note("NeuroSTIAS requires >= 3 detected genes per panel by default and abstains (Unresolved) otherwise; on the "
             "161-gene MERFISH panel no class panel reaches 3 genes, so the default abstains everywhere by design. "
             "neurostias_min_genes2 is a pre-defined sensitivity setting; the argmax arms use >= 2 genes.")
    exp.note("Oligodendrocyte maturation stages are merged into one lineage class because the author labels do not "
             "separate OPCs from immature oligodendrocytes.")
    exp.note("Layer panels are canonical literature markers chosen before running and not derived from DLPFC.")
    cols = ["method", "coverage", "accuracy_resolved", "accuracy_overall", "macro_f1"]
    exp.finish([("Part A - MERFISH hypothalamus vs author cell classes", md_table(a_res[cols])),
                ("Part A - per-class recall", md_table(a_res[["method"] + [c for c in a_res.columns if c.startswith("recall_")]])),
                ("Part B - DLPFC layers vs manual annotation (mean over 12 sections)", md_table(b_sum)),
                ("Part C - Human Brain 3D (descriptive, no ground truth)",
                 "*(excluded from the public release: private dataset)*")])
    print(a_res[cols].to_string()); print(b_sum.to_string()); print(c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a, _ = ap.parse_known_args()
    main(a.quick)
