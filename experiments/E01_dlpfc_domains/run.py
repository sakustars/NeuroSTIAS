"""E01 - Spatial domain (cortical layer) detection on the 12 DLPFC sections.

Methods (same number of domains k = number of manually annotated layers in each
section; spots without a manual label are excluded from scoring):
  neurostias        expression + 2-hop neighbourhood embedding (lam=0.8, k_nn=6, hex grid), k-means, refinement
  neurostias_nosp   ablation: lam=0 (no spatial information)
  neurostias_norefine ablation: no majority-vote refinement
  neurostias_lodo   hyperparameters (lam, hops, k_nn, k-means vs Gaussian mixture) selected by leave-one-donor-out: chosen on the
                    other two donors' sections (seed 0, mean ARI), then evaluated on the held-out donor
  kmeans_pca        non-spatial k-means on 30 PCs
  scanpy_leiden     non-spatial Leiden with resolution search to k
  v441_spatial      3D-STIAS V4.4.1 community analysis (k-means on coordinates)
  v441_combined     V4.4.1 combined mode (coordinates + 10 PCs)
  banksy, spagcn, graphst   published methods, isolated environment, tutorial defaults
Metrics: ARI and NMI per section and seed; summary = mean over seeds per section,
then mean and 95% bootstrap CI over sections; donor-level means reported too.
Hyperparameters of neurostias were fixed before running (BANKSY defaults), not tuned on DLPFC.
"""
import argparse
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neurostias import data  # noqa: E402
from neurostias.core.stats import bootstrap_ci  # noqa: E402
from neurostias.experiment import Experiment, md_table, run_baseline_script  # noqa: E402
from neurostias.legacy.community import community_v441  # noqa: E402
from neurostias.spatial3d.domains import DomainParams, detect_domains  # noqa: E402

SECTIONS = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672",
            "151673", "151674", "151675", "151676"]
BASELINES = ["banksy", "spagcn", "graphst"]


def prep(a):
    a.X = a.layers["counts"].copy()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.layers["log_normalized"] = a.X.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=3000, flavor="seurat")
    h = a[:, a.var.highly_variable].copy()
    sc.pp.scale(h, max_value=10)
    sc.tl.pca(h, n_comps=30, random_state=0)
    a.obsm["X_pca"] = h.obsm["X_pca"]
    return a


def leiden_k(a, k, seed):
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca", random_state=seed)
    lo, hi, lab = 0.01, 3.0, None
    for _ in range(30):
        r = (lo + hi) / 2
        sc.tl.leiden(a, resolution=r, key_added="ld", random_state=seed, flavor="igraph", n_iterations=2, directed=False)
        lab = a.obs["ld"].astype(str).to_numpy()
        n = len(np.unique(lab))
        if n == k:
            break
        lo, hi = (r, hi) if n < k else (lo, r)
    return lab


GRID = [dict(lam=l, hops=h, k_neighbors=k, cluster_method=c) for l in (0.5, 0.8, 0.95) for h in (1, 2, 4, 8)
        for k in (6, 18) for c in ("kmeans", "gmm")]


def lodo_selection(cache: dict) -> tuple[dict, pd.DataFrame]:
    """cache: section -> (adata, truth, mask, k, donor). Returns ({donor: params}, grid table)."""
    rows = []
    for s, (a, truth, m, k, donor) in cache.items():
        for gi, g in enumerate(GRID):
            detect_domains(a, DomainParams(n_domains=k, seed=0, **g), key_added="dom")
            rows.append({"section": s, "donor": donor, "grid_id": gi, **g,
                         "ari": adjusted_rand_score(truth[m], a.obs["dom"].astype(str).to_numpy()[m])})
    grid = pd.DataFrame(rows)
    chosen = {}
    for d in grid["donor"].unique():
        tr = grid[grid.donor != d].groupby("grid_id")["ari"].mean()
        chosen[d] = GRID[int(tr.idxmax())] if len(tr) else dict(lam=0.8, hops=2, k_neighbors=6, cluster_method="kmeans")
    return chosen, grid


def main(quick: bool, skip_baselines: bool):
    exp = Experiment("E01", "Spatial domain detection on human DLPFC (12 Visium sections, 3 donors)",
                     "How accurately do NeuroSTIAS domains recover manually annotated cortical layers compared with "
                     "published spatial-domain methods and the 3D-STIAS V4.4.1 approach?",
                     params={"k_nn": 6, "lam": 0.8, "hops": 2, "n_pcs": 30, "n_hvg": 3000, "refine": True})
    seeds = exp.seeds[:2] if quick else exp.seeds
    sections = ["151507", "151673"] if quick else SECTIONS
    rows = []
    cache = {}
    for s in sections:
        path = data.path("dlpfc", f"{s}.h5ad")
        exp.prov.add_input(path, f"dlpfc_{s}")
        a = prep(sc.read_h5ad(path))
        truth = a.obs["layer"].astype(str).to_numpy()
        m = truth != "NA"
        cache[s] = (a, truth, m, len(np.unique(truth[m])), a.obs["donor"].iloc[0])
    chosen, grid = lodo_selection(cache)
    exp.table(grid, "e01_lodo_grid.csv")
    exp.summary["lodo_selected_params"] = chosen
    for s in sections:
        path = data.path("dlpfc", f"{s}.h5ad")
        a, truth, m, k, donor = cache[s]

        def score(method, seed, lab, sec):
            lab = np.asarray(lab).astype(str)
            rows.append({"section": s, "donor": donor, "method": method, "seed": seed, "k": k,
                         "ari": adjusted_rand_score(truth[m], lab[m]), "nmi": normalized_mutual_info_score(truth[m], lab[m]),
                         "seconds": sec})

        for seed in seeds:
            for name, p in (("neurostias", DomainParams(n_domains=k, seed=seed)),
                            ("neurostias_nosp", DomainParams(n_domains=k, lam=0.0, refine=False, seed=seed)),
                            ("neurostias_norefine", DomainParams(n_domains=k, refine=False, seed=seed)),
                            ("neurostias_lodo", DomainParams(n_domains=k, seed=seed, **chosen[donor]))):
                t = time.time()
                detect_domains(a, p, key_added="dom")
                score(name, seed, a.obs["dom"], time.time() - t)
            t = time.time()
            score("kmeans_pca", seed, KMeans(k, n_init=10, random_state=seed).fit_predict(a.obsm["X_pca"]), time.time() - t)
            t = time.time()
            score("scanpy_leiden", seed, leiden_k(a, k, seed), time.time() - t)
            t = time.time()
            score("v441_spatial", seed, community_v441(a, "spatial", n_clusters=k, seed=seed), time.time() - t)
            t = time.time()
            score("v441_combined", seed, community_v441(a, "combined", n_clusters=k, seed=seed), time.time() - t)
            if not skip_baselines:
                for b in BASELINES:
                    with tempfile.TemporaryDirectory() as td:
                        out = Path(td) / "lab.csv"
                        t = time.time()
                        ok, log = run_baseline_script(ROOT / "benchmarks/baselines/domain_baselines.py",
                                                      ["--input", str(path), "--method", b, "--k", str(k), "--seed", str(seed),
                                                       "--out", str(out)], timeout=3600)
                        if ok:
                            lab = pd.read_csv(out).set_index("obs")["label"].reindex(a.obs_names).fillna("filtered").to_numpy()
                            score(b, seed, lab, time.time() - t)
                            exp.baseline(b, "ran")
                        else:
                            exp.baseline(b, "failed", log)
            print(s, seed, "done", flush=True)
        pd.DataFrame(rows).to_csv(exp.res / "e01_scores_raw_partial.csv", index=False)
    raw = pd.DataFrame(rows)
    exp.table(raw, "e01_scores_raw.csv")
    per_sec = raw.groupby(["method", "section", "donor"], as_index=False)[["ari", "nmi", "seconds"]].mean()
    exp.table(per_sec, "e01_scores_per_section.csv")
    summ = []
    for meth, g in per_sec.groupby("method"):
        ari = bootstrap_ci(g["ari"].to_numpy(), seed=0)
        nmi = bootstrap_ci(g["nmi"].to_numpy(), seed=0)
        summ.append({"method": meth, "n_sections": len(g), "ari_mean": ari[0], "ari_ci_low": ari[1], "ari_ci_high": ari[2],
                     "nmi_mean": nmi[0], "nmi_ci_low": nmi[1], "nmi_ci_high": nmi[2],
                     "median_seconds": g["seconds"].median()})
    summ = pd.DataFrame(summ).sort_values("ari_mean", ascending=False)
    exp.table(summ, "e01_summary.csv")
    donor = per_sec.groupby(["method", "donor"], as_index=False)["ari"].mean().pivot(index="method", columns="donor", values="ari")
    exp.table(donor, "e01_ari_by_donor.csv", index=True)
    # paired comparison of neurostias vs each method across sections (Wilcoxon signed-rank)
    from scipy.stats import wilcoxon
    piv = per_sec.pivot(index="section", columns="method", values="ari")
    tests = []
    for meth in piv.columns:
        if meth == "neurostias" or piv[meth].isna().any():
            continue
        d = piv["neurostias"] - piv[meth]
        p = wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else np.nan
        tests.append({"comparison": f"neurostias - {meth}", "mean_ari_difference": d.mean(), "n_sections": len(d),
                      "sections_neurostias_better": int((d > 0).sum()), "wilcoxon_p": p})
    tests = pd.DataFrame(tests)
    exp.table(tests, "e01_paired_tests.csv")
    import matplotlib.pyplot as plt
    order = summ["method"].tolist()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    data_ = [per_sec.loc[per_sec.method == mth, "ari"] for mth in order]
    ax.boxplot(data_, tick_labels=order, showfliers=False)
    for i, d in enumerate(data_, 1):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(d)), d, s=8, color="k", zorder=3)
    ax.set_ylabel("ARI vs manual layers (per section)")
    ax.set_xticklabels(order, rotation=40, ha="right")
    ax.set_title("E01 DLPFC layer recovery (12 sections, mean of seeds)")
    exp.figure(fig, "e01_ari_boxplot.png")
    exp.summary.update({"summary": summ.to_dict("records"), "paired_tests": tests.to_dict("records"),
                        "quick_mode": quick})
    exp.note("Scores exclude spots without a manual layer label (NA).")
    exp.note("k (number of domains) is set to the number of annotated layers per section for every method; "
             "this is the standard DLPFC benchmark protocol and favours no method.")
    exp.note("No histology images were used by any method (SpaGCN run with histology=False).")
    exp.note("Published DLPFC ARIs for BANKSY, GraphST and SpaGCN are often higher because they use R mclust clustering "
             "and/or histology; here all methods ran on CPU without histology, with Leiden resolution search (BANKSY, "
             "GraphST) or k-means initialisation (SpaGCN), so absolute values are not comparable with those papers.")
    exp.note("neurostias (default) uses fixed a-priori parameters; neurostias_lodo selects parameters on the other "
             "donors only, so its held-out scores are not optimistically biased.")
    exp.finish([("Summary (mean over seeds per section; 95% bootstrap CI over sections)", md_table(summ)),
                ("ARI by donor", md_table(donor.reset_index())),
                ("Paired comparison across sections (Wilcoxon signed-rank)", md_table(tests) if len(tests) else "n/a")])
    print(summ.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    a, _ = ap.parse_known_args()
    main(a.quick, a.skip_baselines)
