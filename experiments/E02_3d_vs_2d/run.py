"""E02 - Does a 3D cross-section graph improve on per-section 2D graphs?

Part A (ground truth: manual layers). DLPFC directly adjacent section pairs (10 um apart;
Maynard et al. 2021). Section B is rigidly registered to A with NeuroSTIAS ICP. Arms:
  2D-separate   each section clustered alone with a 2D graph (as in E01)
  2D-joint      both sections clustered together, graph edges only within sections
  3D-joint      both sections clustered together with cross-section edges (z = 10 um)
2D-joint isolates the effect of cross-section edges from the effect of joint clustering.
Registration QC: fraction of cross-section nearest neighbours that share the manual layer,
before vs after registration.

Part B (label-free): molecular cross-validation by binomial count splitting
(Batson et al. 2019; Neufeld et al. 2023). For each spot the train half is smoothed over a
graph and used to predict the test half; metric = mean Poisson deviance per spot (lower =
better estimate of the spot's true expression). Graphs: none, 2D, 3D isotropic kNN with
z_scale in {0.25, 0.5, 1, 2, 4, 8}, and cross-section kNN. The smoothing weight alpha is
tuned on one random half of the genes and evaluated on the other half. Datasets: GSE214349
(5 mice; z from section labels in um, and the earlier order x 20 um as sensitivity) and the
DLPFC pairs.

Part C (MERFISH, author cell classes). Fraction of a cell's k nearest neighbours that share
its class, for 2D (same section) vs 3D neighbourhoods across adjacent 50 um sections.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neurostias import data  # noqa: E402
from neurostias.core import graph  # noqa: E402
from neurostias.core.stats import bootstrap_ci  # noqa: E402
from neurostias.experiment import Experiment, md_table  # noqa: E402
from neurostias.spatial3d.domains import DomainParams, detect_domains  # noqa: E402
from neurostias.spatial3d.reconstruct import reconstruct  # noqa: E402

PAIRS = [("151507", "151508"), ("151509", "151510"), ("151669", "151670"), ("151671", "151672"),
         ("151673", "151674"), ("151675", "151676")]
UM_PER_PX = 100.0 / 137.0  # Visium spot centre spacing 100 um = 137 px in these full-resolution images
Z_SCALES = [0.25, 0.5, 1, 2, 4, 8]


# ------------------------------------------------------------------ helpers
def load_pair(a_id, b_id):
    parts = []
    for s in (a_id, b_id):
        x = sc.read_h5ad(data.path("dlpfc", f"{s}.h5ad"))
        x.obs_names = [f"{s}_{n}" for n in x.obs_names]
        parts.append(x)
    a = sc.concat(parts, join="inner")
    a.obsm["spatial"] = a.obsm["spatial"] * UM_PER_PX
    a.obs["order"] = (a.obs["section"] == b_id).astype(int)
    a.obs["pair"] = f"{a_id}_{b_id}"
    return a


def lognorm(a):
    a.X = a.layers["counts"].copy()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.layers["log_normalized"] = a.X.copy()
    return a


def layer_agreement(a):
    c = a.obsm["spatial_3d"][:, :2]
    s = a.obs["section"].to_numpy()
    lay = a.obs["layer"].astype(str).to_numpy()
    u = np.unique(s)
    ia, ib = np.flatnonzero(s == u[0]), np.flatnonzero(s == u[1])
    d, nn = cKDTree(c[ib]).query(c[ia], k=1)
    la, lb = lay[ia], lay[ib][nn]
    m = (la != "NA") & (lb != "NA")
    return float(np.mean(la[m] == lb[m])), float(np.median(d))


def part_a(exp, seeds):
    rows, reg = [], []
    for a_id, b_id in PAIRS:
        a = lognorm(load_pair(a_id, b_id))
        for p in (a_id, b_id):
            exp.prov.add_input(data.path("dlpfc", f"{p}.h5ad"), f"dlpfc_{p}")
        # registration QC: centroid-only vs ICP
        audit0 = reconstruct(a, "section", coord_key="spatial", order_key="order", z_spacing=10.0, register=False)
        c0 = a.obsm["spatial_3d"].copy()
        a.obsm["spatial_3d"][:, :2] += 0  # unregistered frame
        ag0, d0 = layer_agreement(a)
        audit = reconstruct(a, "section", coord_key="spatial", order_key="order", z_spacing=10.0, register=True)
        ag1, d1 = layer_agreement(a)
        reg.append({"pair": f"{a_id}_{b_id}", "layer_agreement_unregistered": ag0, "layer_agreement_registered": ag1,
                    "median_nn_distance_unreg_um": d0, "median_nn_distance_reg_um": d1,
                    "rotation_deg": float(audit["rotation_degrees"].dropna().iloc[0]),
                    "overlap_after": float(audit["overlap_after"].dropna().iloc[0])})
        truth = a.obs["layer"].astype(str).to_numpy()
        sec = a.obs["section"].to_numpy()
        for method in ("kmeans", "gmm"):
            for seed in seeds:
                arms = {}
                # 2D-separate
                lab = np.empty(a.n_obs, dtype=object)
                for s in (a_id, b_id):
                    sub = a[sec == s].copy()
                    k = len(np.unique(truth[(sec == s) & (truth != "NA")]))
                    detect_domains(sub, DomainParams(n_domains=k, seed=seed, cluster_method=method), section_key="section")
                    lab[sec == s] = sub.obs["neuro_domain"].astype(str).to_numpy()
                arms["2D-separate"] = lab
                k_joint = len(np.unique(truth[truth != "NA"]))
                detect_domains(a, DomainParams(n_domains=k_joint, z_scale=0.0, seed=seed, cluster_method=method),
                               section_key="section", key_added="d2j")
                arms["2D-joint"] = a.obs["d2j"].astype(str).to_numpy()
                # degree-matched 2D: the 3D-joint graph has ~9.5 neighbours per spot (6 in-plane + cross-section)
                for kk in (9, 10):
                    detect_domains(a, DomainParams(n_domains=k_joint, z_scale=0.0, k_neighbors=kk, seed=seed,
                                                   cluster_method=method), section_key="section", key_added=f"d2j{kk}")
                    arms[f"2D-joint-k{kk}"] = a.obs[f"d2j{kk}"].astype(str).to_numpy()
                detect_domains(a, DomainParams(n_domains=k_joint, z_scale=1.0, cross_section=True, k_between=3, seed=seed,
                                               cluster_method=method), section_key="section", key_added="d3j")
                arms["3D-joint"] = a.obs["d3j"].astype(str).to_numpy()
                for arm, l in arms.items():
                    for s in (a_id, b_id):
                        m = (sec == s) & (truth != "NA")
                        rows.append({"pair": f"{a_id}_{b_id}", "section": s, "cluster_method": method, "seed": seed,
                                     "arm": arm, "ari": adjusted_rand_score(truth[m], l[m])})
        print("pair", a_id, b_id, "done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(reg)


def mcv_block(X, coords, sections, graphs, rng, n_hvg=2000):
    """X: counts (spots x genes, csr). Returns {graph_name: mean deviance on eval genes}."""
    Xtr = X.copy()
    Xtr.data = rng.binomial(Xtr.data.astype(np.int64), 0.5).astype(float)
    Xte = X - Xtr
    lib_tr = np.asarray(Xtr.sum(1)).ravel()
    lib_te = np.asarray(Xte.sum(1)).ravel()
    keep = (lib_tr > 0) & (lib_te > 0)
    mean = np.asarray(Xtr[keep].mean(0)).ravel()
    var = np.asarray(Xtr[keep].power(2).mean(0)).ravel() - mean ** 2
    disp = var / np.maximum(mean, 1e-12)
    hv = np.argsort(np.where(mean > 0.05, disp, -np.inf))[::-1][:n_hvg]
    perm = rng.permutation(hv)
    tune_g, eval_g = perm[: len(perm) // 2], perm[len(perm) // 2:]
    # work only on the selected genes: columns of P_hv are hv, positions mapped below
    pos = {g: i for i, g in enumerate(hv)}
    P = (sparse.diags(1.0 / np.maximum(lib_tr, 1)) @ Xtr)[:, hv].tocsr()  # train-half proportions, selected genes
    out = {}
    Yte = {"tune": Xte[keep][:, tune_g].toarray(), "eval": Xte[keep][:, eval_g].toarray()}
    G_IDX = {"tune": np.array([pos[g] for g in tune_g]), "eval": np.array([pos[g] for g in eval_g])}
    L = lib_te[keep, None]

    def deviance(mu_prop, which):
        lam = np.maximum(mu_prop[:, G_IDX[which]] * L, 1e-10)
        y = Yte[which]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / lam), 0.0) - (y - lam)
        return float(2 * t.sum(1).mean())

    Pk = P[keep]
    for name, W in graphs.items():
        if W is None:
            mu = Pk.toarray()
            out[name] = {"alpha": 0.0, "deviance": deviance(mu, "eval"), "mean_degree": 0.0}
            continue
        Wd = W.tocsr().copy()
        Wd.setdiag(0)
        Wd.eliminate_zeros()
        mean_degree = float(Wd.getnnz(axis=1)[keep].mean())
        Wn = graph.row_normalize(W)
        S = (Wn @ P)[keep].toarray()
        own = Pk.toarray()
        best = None
        for alpha in ALPHAS:
            dv = deviance((1 - alpha) * own + alpha * S, "tune")
            if best is None or dv < best[1]:
                best = (alpha, dv)
        out[name] = {"alpha": best[0], "deviance": deviance((1 - best[0]) * own + best[0] * S, "eval"),
                     "mean_degree": mean_degree}
    return out


ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)  # smoothing weight grid


def graphs_for(coords, sections, k=6):
    blocks = np.repeat("b", len(coords))
    g = {"none": None, "2D": graph.knn_graph(coords, blocks, k=k, z_scale=0.0, sections=sections)}
    # degree-matched 2D controls: the cross-section graph has ~9-12 neighbours per spot, so a gain over
    # 6-neighbour 2D could come from extra smoothing alone
    # dense k grid so every cross-section graph's degree is bracketed within ~1 neighbour (deviance falls convexly
    # with degree, so interpolating between distant 2D points would overstate the 2D deviance)
    for kk in (9, 10, 12, 15, 16, 18, 21, 22, 24):
        g[f"2D_k{kk}"] = graph.knn_graph(coords, blocks, k=kk, z_scale=0.0, sections=sections)
    for z in Z_SCALES:
        g[f"3D_iso_z{z}"] = graph.knn_graph(coords, blocks, k=k, z_scale=z)
    g["3D_cross_section"] = graph.cross_section_knn(coords, sections, blocks, k_within=k, k_between=3, z_scale=1.0)
    # matched-degree family: same cross-section neighbours added to larger in-plane neighbourhoods, so the
    # 2D_k* and 3D_cross_k* curves can be compared at equal mean degree
    for kk in (12, 18):
        g[f"3D_cross_k{kk}"] = graph.cross_section_knn(coords, sections, blocks, k_within=kk, k_between=3, z_scale=1.0)
    return g


def part_b(exp, seeds):
    rows = []
    gse = data.path("gse214349_3d")
    exp.prov.add_input(gse, "GSE214349_reconstructed_3d")
    g = sc.read_h5ad(gse)
    X = g.X.tocsr().astype(float)
    for mouse in g.obs["sample_id"].unique():
        m = (g.obs["sample_id"] == mouse).to_numpy()
        secs = g.obs.loc[m, "slice_id"].astype(str).to_numpy()
        xy = np.asarray(g.obsm["spatial_registered"])[m]
        for zmode in ("label_um", "order_x20um"):
            if zmode == "label_um":
                z = g.obs.loc[m, "section_order"].astype(float).to_numpy()
            else:
                z = np.asarray(g.obsm["spatial_3d"])[m, 2]
            coords = np.column_stack([xy, z])
            G = graphs_for(coords, secs)
            for seed in seeds:
                res = mcv_block(X[m], coords, secs, G, np.random.default_rng(seed))
                for name, r in res.items():
                    rows.append({"dataset": "GSE214349", "block": mouse, "n_sections": len(np.unique(secs)),
                                 "z_mode": zmode, "seed": seed, "graph": name, **r})
        print("mouse", mouse, "done", flush=True)
    for a_id, b_id in PAIRS:
        a = load_pair(a_id, b_id)
        reconstruct(a, "section", coord_key="spatial", order_key="order", z_spacing=10.0, register=True)
        secs = a.obs["section"].to_numpy()
        G = graphs_for(a.obsm["spatial_3d"], secs)
        for seed in seeds:
            res = mcv_block(a.layers["counts"].tocsr().astype(float), a.obsm["spatial_3d"], secs, G,
                            np.random.default_rng(seed))
            for name, r in res.items():
                rows.append({"dataset": "DLPFC_pairs", "block": f"{a_id}_{b_id}", "n_sections": 2, "z_mode": "10um",
                             "seed": seed, "graph": name, **r})
        print("dlpfc pair", a_id, "done", flush=True)
    return pd.DataFrame(rows)


def part_c(exp):
    p = data.path("merfish_hypothalamus")
    exp.prov.add_input(p, "merfish_moffitt2018")
    a = sc.read_h5ad(p)
    keep = a.obs["Cell_class"].astype(str) != "Ambiguous"
    a = a[keep.to_numpy()].copy()
    xy = a.obs[["Centroid_X", "Centroid_Y"]].to_numpy(float)
    z = a.obs["Bregma"].astype(float).to_numpy() * 10.0  # Bregma in units of 0.01 mm -> um
    coords = np.column_stack([xy, z])
    cls = a.obs["Cell_class"].astype(str).str.replace(r" \d+$", "", regex=True).to_numpy()  # merge numbered subclasses
    secs = a.obs["Bregma"].astype(str).to_numpy()
    blocks = np.repeat("animal1", len(coords))
    rows = []
    for k in (5, 10, 20):
        for name, W in (("2D", graph.knn_graph(coords, blocks, k=k, z_scale=0.0, sections=secs, symmetric=False)),
                        ("3D_iso_z1", graph.knn_graph(coords, blocks, k=k, z_scale=1.0, symmetric=False)),
                        ("cross_section_only", None)):
            if W is None:
                # neighbours drawn only from adjacent sections (no same-section neighbours)
                order = sorted(np.unique(z))
                same = []
                for i_z, zz in enumerate(order):
                    src = np.flatnonzero(z == zz)
                    adj = np.flatnonzero(np.isin(z, [order[j] for j in (i_z - 1, i_z + 1) if 0 <= j < len(order)]))
                    _, nn = cKDTree(xy[adj]).query(xy[src], k=k)
                    same.append(np.mean(cls[adj][nn] == cls[src][:, None], axis=1))
                frac = np.concatenate(same)
            else:
                W = W.tocsr()
                frac = np.array([np.mean(cls[W.indices[W.indptr[i]:W.indptr[i + 1]]] == cls[i]) for i in range(W.shape[0])])
            # chance: fraction expected if neighbours were random cells of the same section
            rows.append({"k": k, "neighbourhood": name, "mean_same_class_fraction": float(np.mean(frac))})
    chance = float(np.sum((pd.Series(cls).value_counts(normalize=True)) ** 2))
    return pd.DataFrame(rows), chance


RESUME_A = False
RESUME_B = False


def main(quick):
    exp = Experiment("E02", "3D cross-section graphs versus per-section 2D graphs",
                     "Do neighbours from adjacent serial sections improve spatial-domain recovery and expression "
                     "estimation compared with 2D within-section neighbours, and how should z be weighted?",
                     params={"um_per_px_dlpfc": UM_PER_PX, "z_scales": Z_SCALES, "k_nn": 6, "k_between": 3})
    seeds = exp.seeds[:2] if quick else exp.seeds
    global PAIRS
    if quick:
        PAIRS = PAIRS[:1]
    pa, pr = exp.res / "e02a_dlpfc_pairs_scores.csv", exp.res / "e02a_registration_qc.csv"
    if RESUME_A and pa.exists() and pr.exists():
        a_scores, reg = pd.read_csv(pa), pd.read_csv(pr)  # Part A finished in an earlier run of this script
        exp.note("Part A tables were produced by an earlier invocation of this same script and reused (--resume-a).")
    else:
        a_scores, reg = part_a(exp, seeds)
    exp.table(a_scores, "e02a_dlpfc_pairs_scores.csv")
    exp.table(reg, "e02a_registration_qc.csv")
    per = a_scores.groupby(["cluster_method", "arm", "section"], as_index=False)["ari"].mean()
    summ_a = []
    for (cm, arm), gdf in per.groupby(["cluster_method", "arm"]):
        m, lo, hi = bootstrap_ci(gdf["ari"].to_numpy(), seed=0)
        summ_a.append({"cluster_method": cm, "arm": arm, "n_sections": len(gdf), "ari_mean": m, "ci_low": lo, "ci_high": hi})
    summ_a = pd.DataFrame(summ_a)
    exp.table(summ_a, "e02a_summary.csv")
    from scipy.stats import wilcoxon
    tests = []
    for cm in per["cluster_method"].unique():
        pv = per[per.cluster_method == cm].pivot(index="section", columns="arm", values="ari")
        for other in ("2D-separate", "2D-joint", "2D-joint-k9", "2D-joint-k10"):
            d = pv["3D-joint"] - pv[other]
            tests.append({"cluster_method": cm, "comparison": f"3D-joint - {other}", "mean_diff": d.mean(),
                          "sections_3D_better": int((d > 0).sum()), "n": len(d),
                          "wilcoxon_p": wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else np.nan})
    tests = pd.DataFrame(tests)
    exp.table(tests, "e02a_paired_tests.csv")

    braw = exp.res / "e02b_mcv_raw.csv"
    if RESUME_B and braw.exists():
        b = pd.read_csv(braw)  # Part B finished in an earlier run of this script
        exp.note("Part B raw table was produced by an earlier invocation of this same script and reused (--resume-b).")
    else:
        b = part_b(exp, seeds[:3])  # count-splitting seeds (thinning variance is small)
    exp.table(b, "e02b_mcv_raw.csv")
    rel = b.copy()
    base = rel[rel.graph == "2D"].set_index(["dataset", "block", "z_mode", "seed"])["deviance"]
    rel["deviance_vs_2D_pct"] = 100 * (rel["deviance"] / base.reindex(pd.MultiIndex.from_frame(rel[["dataset", "block", "z_mode", "seed"]])).to_numpy() - 1)
    summ_b = (rel.groupby(["dataset", "z_mode", "graph"], as_index=False)
              .agg(mean_deviance=("deviance", "mean"), deviance_vs_2D_pct=("deviance_vs_2D_pct", "mean"),
                   mean_alpha=("alpha", "mean"), mean_degree=("mean_degree", "mean"), n_blocks=("block", "nunique")))
    exp.table(summ_b, "e02b_mcv_summary.csv")
    matched = []
    for key, gdf in rel.groupby(["dataset", "block", "z_mode", "seed"]):
        two = gdf[gdf.graph.str.match(r"^2D(_k\d+)?$")].sort_values("mean_degree")
        for _, r in gdf[gdf.graph.str.startswith("3D_cross")].iterrows():
            inside = two["mean_degree"].min() <= r["mean_degree"] <= two["mean_degree"].max()
            d2 = np.interp(r["mean_degree"], two["mean_degree"], two["deviance"])
            matched.append({"dataset": key[0], "block": key[1], "z_mode": key[2], "seed": key[3], "graph": r["graph"],
                            "mean_degree": r["mean_degree"], "deviance_3d": r["deviance"],
                            "deviance_2d_at_same_degree": d2, "within_2d_degree_range": bool(inside),
                            "pct_vs_degree_matched_2D": 100 * (r["deviance"] / d2 - 1)})
    matched = pd.DataFrame(matched)
    from scipy.stats import wilcoxon as _wx
    mt = []
    for (ds, zm, gname), gdf in matched.groupby(["dataset", "z_mode", "graph"]):
        blk = gdf.groupby("block")[["deviance_3d", "deviance_2d_at_same_degree", "pct_vs_degree_matched_2D", "mean_degree"]].mean()
        diff = blk["deviance_3d"] - blk["deviance_2d_at_same_degree"]
        mt.append({"dataset": ds, "z_mode": zm, "graph": gname, "mean_degree": blk["mean_degree"].mean(),
                   "pct_vs_degree_matched_2D": blk["pct_vs_degree_matched_2D"].mean(),
                   "blocks_3D_better": int((diff < 0).sum()), "n_blocks": len(blk),
                   "within_2d_degree_range": bool(gdf["within_2d_degree_range"].all()),
                   "wilcoxon_p": _wx(diff).pvalue if len(diff) >= 5 and np.any(diff != 0) else np.nan})
    mt = pd.DataFrame(mt)
    exp.table(matched, "e02b_degree_matched_raw.csv")
    exp.table(mt, "e02b_degree_matched.csv")
    best = summ_b[summ_b.graph.str.startswith("3D")].sort_values("deviance_vs_2D_pct").groupby(["dataset", "z_mode"]).head(1)
    exp.table(best, "e02b_best_3d_graph.csv")

    c, chance = part_c(exp)
    exp.table(c, "e02c_merfish_class_coherence.csv")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for cm, ax in zip(("kmeans", "gmm"), axes):
        pv = per[per.cluster_method == cm].pivot(index="section", columns="arm", values="ari")[["2D-separate", "2D-joint", "2D-joint-k10", "3D-joint"]]
        for _, r in pv.iterrows():
            ax.plot(range(4), r.values, color="grey", alpha=0.6, marker="o", ms=3)
        ax.plot(range(4), pv.mean().values, color="k", lw=2, marker="o")
        ax.set_xticks(range(4), pv.columns, fontsize=7)
        ax.set_ylabel("ARI vs manual layers")
        ax.set_title(f"DLPFC adjacent pairs ({cm})")
    exp.figure(fig, "e02a_pairs_ari.png")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for (ds, zm), gdf in summ_b.groupby(["dataset", "z_mode"]):
        iso = gdf[gdf.graph.str.startswith("3D_iso")].copy()
        iso["z"] = iso.graph.str.replace("3D_iso_z", "").astype(float)
        iso = iso.sort_values("z")
        ax.plot(iso["z"], iso["deviance_vs_2D_pct"], marker="o", label=f"{ds} ({zm})")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("z_scale (multiplier on inter-section distance)")
    ax.set_ylabel("Held-out deviance vs 2D (%)\n(negative = 3D better)")
    ax.legend(frameon=False, fontsize=6)
    exp.figure(fig, "e02b_mcv_zscale.png")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, ((ds, zm), gdf) in zip(axes, summ_b.groupby(["dataset", "z_mode"])):
        for fam, pref, col in (("2D in-plane", "2D", "tab:blue"), ("3D cross-section", "3D_cross", "tab:red")):
            f = gdf[gdf.graph.str.startswith(pref) & ~gdf.graph.str.startswith("3D_iso")].sort_values("mean_degree")
            ax.plot(f["mean_degree"], f["mean_deviance"], marker="o", color=col, label=fam)
        ax.set_xlabel("mean neighbours per spot")
        ax.set_ylabel("held-out deviance")
        ax.set_title(f"{ds} ({zm})", fontsize=8)
        ax.legend(frameon=False, fontsize=6)
    exp.figure(fig, "e02b_deviance_vs_degree.png")

    exp.summary.update({"part_a_summary": summ_a.to_dict("records"), "part_a_tests": tests.to_dict("records"),
                        "registration_qc": reg.to_dict("records"), "part_b_best_3d": best.to_dict("records"),
                        "part_b_degree_matched": mt.to_dict("records"),
                        "part_c": c.to_dict("records"), "part_c_chance_same_class": chance, "quick_mode": quick})
    exp.note("DLPFC pixel-to-um conversion assumes the 100 um Visium spot pitch (137 px).")
    exp.note("GSE214349 z: section labels interpreted as positions in um (GEO: 20 um intervals, 10 um error); "
             "the previous order x 20 um reconstruction is reported as a sensitivity analysis.")
    exp.note("Part B tunes the smoothing weight on half of the genes and evaluates on the other half, so graphs are "
             "compared at their own best weight without using the evaluation genes.")
    exp.note("Part B includes degree-matched 2D controls because the cross-section "
             "graph has more neighbours per spot than the 6-neighbour 2D graph; 2D graphs with k = 6, 9, 10, 12, 15, 16, 18, 21, "
             "22 and 24 bracket every cross-section graph's degree within about one neighbour. The 'degree-matched' table compares each "
             "cross-section graph with the 2D deviance-vs-degree curve linearly interpolated at the same mean degree "
             "(per block and seed; Wilcoxon over blocks).")
    exp.note("Interpretation: the 3D graphs have more neighbours per spot than the 6-neighbour 2D graph, so their effect "
             "must be read against the degree-matched 2D arms (Part A: 2D-joint-k9/k10; Part B: degree-matched table). "
             "Comparisons with the 6-neighbour 2D graph alone confound cross-section information with neighbourhood size.")
    exp.note("Part A degree-matched arms (2D-joint-k9, 2D-joint-k10) were added after the first full run showed the "
             "neighbourhood-size confound in Part B; they are reported alongside the original arms.")
    exp.note("Smoothing weight grid: 0-0.9 in steps of 0.1 plus 0.95 and 1.0 (1.0 = neighbour average only).")
    exp.note("MERFISH values are volume-normalised (not counts), so it is used only in Part C.")
    exp.note("Part B uses 3 count-splitting seeds per block; Part A uses 5 clustering seeds.")
    exp.finish([("Part A - DLPFC adjacent pairs, ARI per section (mean over seeds; 95% bootstrap CI)", md_table(summ_a)),
                ("Part A - paired tests across sections", md_table(tests)),
                ("Part A - registration QC (cross-section nearest neighbour shares manual layer)", md_table(reg)),
                ("Part B - molecular cross-validation (mean over blocks and seeds)", md_table(summ_b)),
                ("Part B - best 3D graph per dataset", md_table(best)),
                ("Part B - cross-section graphs versus 2D at the same mean degree (negative = 3D better)", md_table(mt)),
                ("Part C - MERFISH neighbour class coherence", md_table(c) + f"\n\nChance level (random pairs): {chance:.3f}")])
    print(summ_a.to_string()); print(tests.to_string()); print(best.to_string()); print(c.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume-a", action="store_true")
    ap.add_argument("--resume-b", action="store_true")
    a, _ = ap.parse_known_args()
    RESUME_A = a.resume_a
    RESUME_B = a.resume_b
    main(a.quick)
