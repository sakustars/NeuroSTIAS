"""E05 - Pseudoreplication: false-positive control of spot-level vs replicate-aware DE.

Part A (null, real data). DLPFC: restrict to one manually annotated layer (L3, L5, WM), split
the 12 sections into two groups balanced within donor (2 + 2 sections per donor), and test
group A vs B. The split carries no biology, so every gene called significant is a false
positive. Methods:
  wilcoxon_spot      Scanpy rank_genes_groups (Wilcoxon) on spots - the common default
  ttest_spot_v441    observation-level Welch t-test with BH (3D-STIAS V4.4.1 DE)
  pseudobulk_section pseudobulk per section, Welch t across sections (6 vs 6)
  pseudobulk_donor   pseudobulk per donor x group, paired t across the 3 donors
Metrics: genes with BH-FDR < 0.05, fraction of splits with >= 1 false discovery, and the
fraction of nominal p < 0.05 (should be ~0.05 when calibrated).

Part B (power; semi_synthetic). Same null splits, plus a known effect: for 200 random
expressed genes, counts in group B are binomially thinned to 50% (true log2FC = 1 for A vs B).
Metrics: sensitivity at BH-FDR < 0.05 and observed false discovery proportion.

Part C (design audit, real data). GSE214349 IVH day 1 vs no-surgery control: one mouse per
condition. Spot-level Wilcoxon reports many genes; replicate-aware testing correctly
reports that no p-value is possible.
"""
import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neurostias import data  # noqa: E402
from neurostias.core.stats import bh, pseudobulk_test  # noqa: E402
from neurostias.experiment import Experiment, md_table  # noqa: E402

DONORS = {"Br5292": ["151507", "151508", "151509", "151510"], "Br5595": ["151669", "151670", "151671", "151672"],
          "Br8100": ["151673", "151674", "151675", "151676"]}
LAYERS = ["L3", "L5", "WM"]


def balanced_splits(n, rng):
    per = [list(itertools.combinations(v, 2)) for v in DONORS.values()]
    allc = list(itertools.product(*per))
    idx = rng.choice(len(allc), size=min(n, len(allc)), replace=False)
    return [set(s for pair in allc[i] for s in pair) for i in idx]


def spot_tests(X_log, grp, genes):
    a, b = grp == "A", grp == "B"
    out = {}
    U = stats.mannwhitneyu(X_log[a], X_log[b], axis=0, alternative="two-sided")
    out["wilcoxon_spot"] = np.nan_to_num(U.pvalue, nan=1.0)
    t = stats.ttest_ind(X_log[a], X_log[b], axis=0, equal_var=False)
    out["ttest_spot_v441"] = np.nan_to_num(t.pvalue, nan=1.0)
    return out


def pb_tests(counts, grp, section, donor, genes):
    out = {}
    r = pseudobulk_test(counts, genes, pd.Series(grp), pd.Series(section), "A", "B", min_cells=10, paired=False)
    out["pseudobulk_section"] = r["pval"].fillna(1.0).to_numpy()
    r = pseudobulk_test(counts, genes, pd.Series(grp), pd.Series(donor), "A", "B", min_cells=10, paired=True)
    out["pseudobulk_donor"] = r["pval"].fillna(1.0).to_numpy()
    return out


def load_dlpfc():
    a = sc.read_h5ad(data.path("dlpfc"))
    a.X = a.layers["counts"].tocsr().astype(float)
    return a


def run_split(sub, groups_secs, genes_idx, effect_genes=None, rng=None):
    sec = sub.obs["section"].astype(str).to_numpy()
    grp = np.where(np.isin(sec, list(groups_secs)), "A", "B")
    C = sub.X[:, genes_idx].tocsr().copy()
    if effect_genes is not None:
        C = C.tolil()
        bmask = np.flatnonzero(grp == "B")
        Cc = C.tocsc()
        for g in effect_genes:
            col = Cc[:, g].toarray().ravel()
            col[bmask] = rng.binomial(col[bmask].astype(np.int64), 0.5)
            Cc[:, g] = col[:, None]
        C = Cc.tocsr()
    lib = np.asarray(sub.X.sum(1)).ravel()
    Xl = np.log1p(C.multiply(1e4 / lib[:, None]).toarray())
    genes = np.arange(len(genes_idx)).astype(str)
    p = spot_tests(Xl, grp, genes)
    p.update(pb_tests(C, grp, sec, sub.obs["donor"].astype(str).to_numpy(), genes))
    return p


def main(quick):
    exp = Experiment("E05", "Pseudoreplication and false-positive control in spatial differential expression",
                     "Do spot-level tests (the common default and 3D-STIAS V4.4.1) produce false discoveries when "
                     "groups differ only by section/donor, and does replicate-aware pseudobulk testing control them?",
                     params={"layers": LAYERS, "n_effect_genes": 200, "thinning": 0.5, "min_cells": 10})
    rng = np.random.default_rng(0)
    n_splits = 4 if quick else 30
    a = load_dlpfc()
    exp.prov.add_input(data.path("dlpfc"), "dlpfc_all")
    rows, power_rows = [], []
    for layer in (LAYERS[:1] if quick else LAYERS):
        sub = a[a.obs["layer"].astype(str) == layer].copy()
        frac = np.asarray((sub.X > 0).mean(0)).ravel()
        genes_idx = np.flatnonzero(frac >= 0.05)
        for si, secs in enumerate(balanced_splits(n_splits, rng)):
            p = run_split(sub, secs, genes_idx)
            for meth, pv in p.items():
                q = bh(pv)
                rows.append({"layer": layer, "split": si, "method": meth, "n_genes": len(pv),
                             "n_fdr05": int((q < 0.05).sum()), "frac_p05": float(np.mean(pv < 0.05))})
            eff = rng.choice(len(genes_idx), size=200, replace=False)
            p2 = run_split(sub, secs, genes_idx, effect_genes=eff, rng=rng)
            truth = np.zeros(len(genes_idx), bool)
            truth[eff] = True
            for meth, pv in p2.items():
                q = bh(pv)
                called = q < 0.05
                power_rows.append({"layer": layer, "split": si, "method": meth,
                                   "sensitivity": float(called[truth].mean()),
                                   "fdp": float((called & ~truth).sum() / max(called.sum(), 1)),
                                   "n_called": int(called.sum())})
            print(layer, si, flush=True)
    null = pd.DataFrame(rows)
    power = pd.DataFrame(power_rows)
    exp.table(null, "e05a_null_raw.csv")
    exp.table(power, "e05b_power_raw_semi_synthetic.csv")
    s_null = null.groupby("method", as_index=False).agg(
        median_false_discoveries=("n_fdr05", "median"), mean_false_discoveries=("n_fdr05", "mean"),
        frac_splits_with_any_fd=("n_fdr05", lambda x: float(np.mean(x > 0))), mean_frac_p_below_005=("frac_p05", "mean"),
        n_tests=("n_fdr05", "size"))
    s_pow = power.groupby("method", as_index=False).agg(mean_sensitivity=("sensitivity", "mean"),
                                                         mean_fdp=("fdp", "mean"), median_called=("n_called", "median"))
    exp.table(s_null, "e05a_null_summary.csv")
    exp.table(s_pow, "e05b_power_summary_semi_synthetic.csv")

    # Part C: one mouse per condition in GSE214349
    g = sc.read_h5ad(data.path("gse214349_3d"))
    exp.prov.add_input(data.path("gse214349_3d"), "GSE214349_reconstructed_3d")
    sel = g.obs["condition"].astype(str).isin(["IVH", "no-surgery control"]) & g.obs["timepoint"].astype(str).isin(["Day 1", "Day 0"])
    sub = g[sel.to_numpy()].copy()
    sub.var_names = sub.var["gene_symbol"].astype(str) if "gene_symbol" in sub.var else sub.var_names
    sub.var_names_make_unique()
    frac = np.asarray((sub.X > 0).mean(0)).ravel()
    sub = sub[:, frac >= 0.05].copy()
    grp = np.where(sub.obs["condition"].astype(str) == "IVH", "A", "B")
    C = sub.X.tocsr().astype(float)
    lib = np.asarray(C.sum(1)).ravel()
    Xl = np.log1p(C.multiply(1e4 / lib[:, None]).toarray())
    pw = stats.mannwhitneyu(Xl[grp == "A"], Xl[grp == "B"], axis=0).pvalue
    r = pseudobulk_test(C, sub.var_names, pd.Series(grp), pd.Series(sub.obs["sample_id"].astype(str).to_numpy()), "A", "B")
    part_c = {"n_spots_ivh_d1": int((grp == "A").sum()), "n_spots_control": int((grp == "B").sum()),
              "n_mice_per_group": 1, "n_genes_tested": int(sub.n_vars),
              "wilcoxon_spot_n_fdr05": int((bh(np.nan_to_num(pw, nan=1)) < 0.05).sum()),
              "pseudobulk_result": r["test"].iloc[0]}
    exp.summary.update({"part_a_null": s_null.to_dict("records"), "part_b_power_semi_synthetic": s_pow.to_dict("records"),
                        "part_c_design_audit": part_c, "quick_mode": quick})

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    order = ["wilcoxon_spot", "ttest_spot_v441", "pseudobulk_section", "pseudobulk_donor"]
    axes[0].boxplot([null.loc[null.method == m, "n_fdr05"] + 1 for m in order], tick_labels=order, showfliers=False)
    axes[0].set_yscale("log"); axes[0].set_ylabel("False discoveries + 1 (BH 5%)")
    axes[0].set_title("Null section splits (no true effect)")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(order, [s_pow.set_index("method").loc[m, "mean_sensitivity"] for m in order], color="#0072B2")
    for i, m in enumerate(order):
        axes[1].text(i, 0.02, f"FDP {s_pow.set_index('method').loc[m, 'mean_fdp']:.2f}", ha="center", color="w", fontsize=7)
    axes[1].set_ylabel("Sensitivity (semi-synthetic)"); axes[1].set_title("Known log2FC=1 in 200 genes")
    axes[1].tick_params(axis="x", rotation=35)
    exp.figure(fig, "e05_null_and_power.png")
    exp.note("Part B is semi-synthetic: real counts with a known effect added by binomial thinning; it is labelled so in "
             "every file name.")
    exp.note("Splits are balanced within donor so donor effects are not confounded with group; remaining false "
             "positives from spot-level tests come from section-level variation treated as replication.")
    exp.note("Genes expressed in at least 5% of spots of the layer are tested.")
    exp.finish([("Part A - null splits (real data; all discoveries are false)", md_table(s_null)),
                ("Part B - power with a known effect (semi-synthetic)", md_table(s_pow)),
                ("Part C - design audit, GSE214349 IVH day 1 vs control (one mouse each)",
                 "\n".join(f"- {k}: {v}" for k, v in part_c.items()))])
    print(s_null.to_string()); print(s_pow.to_string()); print(part_c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a, _ = ap.parse_known_args()
    main(a.quick)
