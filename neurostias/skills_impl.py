"""Skill entry points: thin wrappers that load data, call library functions, and save
tables, figures and (optionally) the updated AnnData into the run folder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import plotting
from .core.io import coords as _coords
from .core.io import read_h5ad, summarize


def _load(ctx, input):
    adata = read_h5ad(input)
    ctx.log.info("loaded %s (%d obs x %d vars)", input, adata.n_obs, adata.n_vars)
    return adata


def _save(ctx, adata, save_h5ad: bool, name: str = "output.h5ad"):
    if save_h5ad:
        p = ctx.path(name)
        for k in list(adata.uns):
            if k.endswith("_colors"):
                del adata.uns[k]
        adata.write_h5ad(p)
        return str(p)
    return None


def _sections(adata, section_key):
    return adata.obs[section_key].astype(str).to_numpy() if section_key and section_key in adata.obs else None


# ----------------------------------------------------------------------------- core
def load_triplet_skill(ctx, expression, coordinates, metadata):
    from .core.io import load_triplet
    adata = load_triplet(expression, coordinates, metadata)
    for label, p in (("expression", expression), ("coordinates", coordinates), ("metadata", metadata)):
        ctx.provenance.add_input(p, label=label)
    out = _save(ctx, adata, True, "loaded.h5ad")
    return {"summary": summarize(adata), "h5ad": out}


def preprocess_skill(ctx, input, min_genes=200, min_cells=3, max_pct_mt=20.0, n_top_genes=2000, n_pcs=50,
                     n_neighbors=15, leiden_resolution=1.0):
    from .core.preprocess import PreprocessParams, preprocess
    adata = _load(ctx, input)
    adata, report = preprocess(adata, PreprocessParams(min_genes=min_genes, min_cells=min_cells, max_pct_mt=max_pct_mt,
                                                       n_top_genes=n_top_genes, n_pcs=n_pcs, n_neighbors=n_neighbors,
                                                       leiden_resolution=leiden_resolution, seed=ctx.seed))
    if "X_umap" in adata.obsm:
        ctx.save_figure(plotting.spatial_panels(adata.obsm["X_umap"], adata.obs["leiden"].astype(str).to_numpy(),
                                                title="UMAP - Leiden clusters"), "umap_leiden.png")
    report["h5ad"] = _save(ctx, adata, True, "preprocessed.h5ad")
    return report


# ----------------------------------------------------------------------------- atlas
def annotate_skill(ctx, input, species="auto", alpha=0.05, min_genes=3, n_null=1000, section_key=None,
                   save_h5ad=True):
    from .atlas.annotate import annotate
    adata = _load(ctx, input)
    res = annotate(adata, species=species, alpha=alpha, min_genes=min_genes, n_null=n_null, seed=ctx.seed)
    ctx.save_table(res["coverage"], "panel_coverage.csv", index=False)
    cols = ["neuro_class", "neuro_subclass", "neuro_transmitter", "neuro_class_margin"] + \
           [c for c in adata.obs.columns if c.startswith("state_")]
    ctx.save_table(adata.obs[cols], "annotations.csv")
    ctx.save_table(res["class_effect"], "class_effect_scores.csv")
    ctx.save_figure(plotting.barh_counts(adata.obs["neuro_class"].value_counts(), "Brain cell classes"),
                    "class_counts.png")
    try:
        c = _coords(adata)
        ctx.save_figure(plotting.spatial_panels(c, adata.obs["neuro_class"].astype(str).to_numpy(),
                                                _sections(adata, section_key), title="Brain cell classes"),
                        "class_spatial.png")
    except KeyError:
        pass
    return {"summary": res["summary"], "h5ad": _save(ctx, adata, save_h5ad)}


# ----------------------------------------------------------------------------- spatial 3D
def reconstruct_skill(ctx, input, section_key, block_key=None, coord_key="spatial", z_key=None, z_spacing=None,
                      order_key=None, register=True, save_h5ad=True):
    from .spatial3d.reconstruct import reconstruct
    adata = _load(ctx, input)
    audit = reconstruct(adata, section_key=section_key, block_key=block_key, coord_key=coord_key, z_key=z_key,
                        z_spacing=z_spacing, order_key=order_key, register=register)
    ctx.save_table(audit.drop(columns=["affine"]), "registration_audit.csv", index=False)
    return {"n_sections": int(len(audit)),
            "median_overlap_after": float(audit["overlap_after"].median()) if "overlap_after" in audit else None,
            "h5ad": _save(ctx, adata, save_h5ad)}


def domains_skill(ctx, input, n_domains=7, z_scale=0.0, cross_section=False, k_neighbors=6, lam=0.8, hops=2,
                  section_key=None, block_key=None, refine=True, save_h5ad=True):
    from .spatial3d.domains import DomainParams, detect_domains
    adata = _load(ctx, input)
    res = detect_domains(adata, DomainParams(n_domains=n_domains or None, z_scale=z_scale, cross_section=cross_section,
                                             k_neighbors=k_neighbors, lam=lam, hops=hops, refine=refine, seed=ctx.seed),
                         section_key=section_key, block_key=block_key)
    ctx.save_table(adata.obs[["neuro_domain"]], "domains.csv")
    ctx.save_figure(plotting.spatial_panels(_coords(adata), adata.obs["neuro_domain"].astype(str).to_numpy(),
                                            _sections(adata, section_key), title="Spatial domains"), "domains.png")
    return {**res, "domain_sizes": adata.obs["neuro_domain"].value_counts().to_dict(), "h5ad": _save(ctx, adata, save_h5ad)}


def svg_skill(ctx, input, n_top_genes=500, k=8, z_scale=1.0, n_permutations=199, section_key=None, block_key=None):
    from .spatial3d.svg import spatially_variable_genes
    adata = _load(ctx, input)
    res = spatially_variable_genes(adata, n_top_genes=n_top_genes, k=k, z_scale=z_scale, n_permutations=n_permutations,
                                   block_key=block_key, section_key=section_key, seed=ctx.seed)
    ctx.save_table(res, "spatially_variable_genes.csv", index=False)
    top = res.head(20).set_index("gene")["morans_i"]
    ctx.save_figure(plotting.barh_counts(top, "Top spatially variable genes", "Moran's I"), "top_svg.png")
    return {"n_tested": int(len(res)), "n_fdr_0.05": int((res["padj"] < 0.05).sum()),
            "top10": res.head(10)["gene"].tolist(), "block_key": res.attrs.get("block_key")}


def niches_skill(ctx, input, label_key="neuro_class", n_niches=6, k=15, z_scale=1.0, section_key=None, block_key=None,
                 save_h5ad=True):
    from .spatial3d.niches import niches
    adata = _load(ctx, input)
    table = niches(adata, label_key, n_niches=n_niches, k=k, z_scale=z_scale, block_key=block_key, seed=ctx.seed)
    ctx.save_table(table, "niche_composition.csv")
    mean_comp = table.groupby("spatial_niche").mean(numeric_only=True)
    ctx.save_figure(plotting.heatmap(mean_comp, "Mean neighbourhood composition per niche", cmap="viridis", center=None),
                    "niche_composition.png")
    ctx.save_figure(plotting.spatial_panels(_coords(adata), adata.obs["spatial_niche"].astype(str).to_numpy(),
                                            _sections(adata, section_key), title="Spatial niches"), "niches.png")
    return {"niche_sizes": adata.obs["spatial_niche"].value_counts().to_dict(), "h5ad": _save(ctx, adata, save_h5ad)}


def communication_skill(ctx, input, label_key="neuro_class", database="neuro", k=8, z_scale=1.0, n_permutations=200,
                        section_key=None, block_key=None, min_replicate_fraction=0.5):
    from .spatial3d.communication import communication
    adata = _load(ctx, input)
    res = communication(adata, label_key, database=database, k=k, z_scale=z_scale, block_key=block_key,
                        section_key=section_key, n_permutations=n_permutations,
                        min_replicate_fraction=min_replicate_fraction, seed=ctx.seed)
    ctx.save_table(res, "communication.csv", index=False)
    out = {"n_tested": int(len(res)), "database": database}
    if len(res):
        sig = res[res["significant"]]
        out.update({"n_significant": int(len(sig)), "n_blocks": res.attrs.get("n_blocks"),
                    "top": sig.head(10)[["interaction", "sender", "receiver", "mean_score", "padj",
                                         "replicate_fraction"]].to_dict("records")})
        if len(sig):
            mat = sig.groupby(["sender", "receiver"]).size().unstack(fill_value=0)
            ctx.save_figure(plotting.heatmap(mat, "Significant interactions (sender rows -> receiver columns)",
                                             cmap="viridis", center=None), "communication_counts.png")
    return out


# ----------------------------------------------------------------------------- statistics / DE
def replicate_de_skill(ctx, input, group_key, group_a, group_b, replicate_key, min_cells=10, paired=True):
    from .legacy.de import replicate_de
    adata = _load(ctx, input)
    res = replicate_de(adata, group_key, group_a, group_b, replicate_key, min_cells=min_cells, paired=paired)
    ctx.save_table(res, "pseudobulk_de.csv", index=False)
    return {"test": res["test"].iloc[0], "n_units": int(res["n_units"].iloc[0]),
            "n_padj_0.05": int((res["padj"] < 0.05).sum()) if res["padj"].notna().any() else 0}


def region_de_v441_skill(ctx, input, n_regions=8, min_cells=10):
    from .legacy.de import region_de_v441
    adata = _load(ctx, input)
    summary, tables = region_de_v441(adata, n_regions=n_regions, min_cells=min_cells, seed=42)
    ctx.save_table(summary, "de_summary.csv", index=False)
    for name, df in tables.items():
        ctx.save_table(df, f"tables/{name}.csv", index=False)
    return {"n_comparisons": int(len(summary)), "n_significant_total": int(summary["n_significant"].sum()),
            "warning": "observation-level test with data-defined regions (V4.4.1 behaviour); see E05"}


# ----------------------------------------------------------------------------- legacy analyses
def grn_skill(ctx, input, tf_set="neural", n_tfs=20, n_targets=50, corr_threshold=0.2, context_key=None):
    from .legacy.grn import tf_coexpression_network
    adata = _load(ctx, input)
    r = tf_coexpression_network(adata, n_tfs=n_tfs, n_targets=n_targets, corr_threshold=corr_threshold,
                                context_key=context_key, tf_set=tf_set)
    if r["status"] == "completed":
        ctx.save_table(r["edges"], "grn_edges.csv", index=False)
        ctx.save_table(r["tests"], "grn_tests.csv", index=False)
    return {k: v for k, v in r.items() if k not in ("edges", "tests")}


def development_skill(ctx, input, species="auto", include_v441_stemness=True, save_h5ad=True, section_key=None):
    from .legacy.development import neurodevelopment_stages, stemness_v441
    adata = _load(ctx, input)
    r = neurodevelopment_stages(adata, species=species, seed=ctx.seed)
    ctx.save_table(r.pop("coverage"), "neurodev_panel_coverage.csv", index=False)
    cols = [c for c in adata.obs.columns if c.startswith(("devstage_", "maturation_index", "neurodev_stage"))]
    out = {"neurodevelopment": r}
    if include_v441_stemness:
        out["stemness_v441"] = stemness_v441(adata)
        cols += [c for c in adata.obs.columns if c.startswith("stemness_")]
    ctx.save_table(adata.obs[cols], "development_scores.csv")
    if "maturation_index" in adata.obs:
        try:
            ctx.save_figure(plotting.spatial_panels(_coords(adata), adata.obs["maturation_index"].to_numpy(),
                                                    _sections(adata, section_key), title="Neuronal maturation index",
                                                    categorical=False), "maturation_index.png")
        except KeyError:
            pass
    out["h5ad"] = _save(ctx, adata, save_h5ad)
    return out


def trajectory_skill(ctx, input, root_label_key=None, root_label=None, root_gene=None, n_neighbors=15, save_h5ad=True):
    from .legacy.trajectory import pseudotime
    adata = _load(ctx, input)
    r = pseudotime(adata, root_label_key=root_label_key, root_label=root_label, root_gene=root_gene,
                   n_neighbors=n_neighbors, seed=ctx.seed)
    ctx.save_table(adata.obs[["pseudotime", "pseudotime_sensitivity_iqr"]], "pseudotime.csv")
    r["h5ad"] = _save(ctx, adata, save_h5ad)
    return r


def ml_skill(ctx, input, label_key, group_key=None, n_label_permutations=20, use_coordinates=True):
    from .legacy.ml import predict_cell_types
    adata = _load(ctx, input)
    return predict_cell_types(adata, label_key, group_key=group_key, n_label_permutations=n_label_permutations,
                              use_coordinates=use_coordinates, seed=ctx.seed)


def enrichment_skill(ctx, de_table, library="neuro_v1", species="human", gene_col="gene", score_col="log2fc",
                     padj_col="padj", padj_threshold=0.05, n_permutations=1000):
    from .legacy.enrichment import get_library, gsea_preranked, ora
    ctx.provenance.add_input(de_table, label="de_table")
    de = pd.read_csv(de_table)
    lib = get_library(library, species)
    sig = de[(de[padj_col] < padj_threshold) & (de[score_col] > 0)][gene_col]
    o = ora(sig, de[gene_col], lib)
    g = gsea_preranked(de.set_index(gene_col)[score_col], lib, n_permutations=n_permutations, seed=ctx.seed)
    ctx.save_table(o, "ora.csv", index=False)
    ctx.save_table(g, "gsea.csv", index=False)
    return {"n_query_genes": int(len(sig)), "ora_top": o.head(5).to_dict("records") if len(o) else [],
            "gsea_top": g.head(5).to_dict("records") if len(g) else []}


def misc_skill(ctx, input, analysis, group_key=None, save_h5ad=False):
    from .legacy import misc
    adata = _load(ctx, input)
    if analysis == "deconvolution":
        P = misc.deconvolve_nnls(adata)
        ctx.save_table(P, "deconvolution_proportions.csv")
        out = {"mean_proportions": P.mean().sort_values(ascending=False).to_dict()}
    elif analysis == "gcn":
        misc.gcn_embedding(adata, seed=ctx.seed)
        ctx.save_table(adata.obs[["spatial_gcn_cluster"]], "gcn_clusters.csv")
        out = {"cluster_sizes": adata.obs["spatial_gcn_cluster"].value_counts().to_dict()}
    elif analysis == "metabolic":
        r = misc.metabolic_scores(adata, group_key=group_key)
        ctx.save_table(r["coverage"], "metabolic_coverage.csv", index=False)
        if r["status"] == "completed":
            ctx.save_table(r["scores"], "metabolic_scores.csv")
            if "by_group" in r:
                ctx.save_table(r["by_group"], "metabolic_by_group.csv")
                ctx.save_figure(plotting.heatmap(r["by_group"], f"Metabolic programmes by {group_key}"), "metabolic.png")
        out = {"status": r["status"]}
    elif analysis == "colocalization":
        r = misc.colocalization_candidates(adata)
        ctx.save_table(r["morans_i"], "morans_i.csv", index=False)
        ctx.save_table(r["pairs"], "candidate_pairs.csv", index=False)
        out = {"n_candidate_pairs": int(len(r["pairs"])), "is_significance_test": False}
    elif analysis == "intervention":
        out = misc.virtual_intervention(adata)
    elif analysis == "community_v441":
        from .legacy.community import community_v441
        community_v441(adata)
        ctx.save_table(adata.obs[["spatial_community"]], "communities.csv")
        out = {"sizes": adata.obs["spatial_community"].value_counts().to_dict()}
    elif analysis == "overview":
        c = _coords(adata)
        key = next((k for k in ("neuro_class", "cell_type", "leiden") if k in adata.obs), None)
        vals = adata.obs[key].astype(str).to_numpy() if key else np.repeat("all", adata.n_obs)
        sec = next((k for k in ("section", "slice_id") if k in adata.obs), None)
        ctx.save_figure(plotting.spatial_panels(c, vals, _sections(adata, sec), title=f"Tissue overview ({key})"),
                        "overview.png")
        out = {"label_key": key, "summary": summarize(adata)}
    else:
        raise ValueError(f"unknown analysis {analysis}")
    out["h5ad"] = _save(ctx, adata, save_h5ad)
    return out


# ----------------------------------------------------------------------------- organoids
def organoid_skill(ctx, input, reference, label_key, region_key=None, age_key=None, cluster_key="leiden",
                   save_h5ad=True):
    from .legacy.development import neurodevelopment_stages
    from .organoid import fidelity as fid
    q = _load(ctx, input)
    ctx.provenance.add_input(reference, label="reference")
    ref = read_h5ad(reference)
    out = {"mapping": fid.map_to_reference(q, ref, label_key, age_key=age_key, seed=ctx.seed)}
    if region_key and cluster_key in q.obs:
        reg = fid.region_identity(q, ref, region_key, cluster_key)
        ctx.save_table(reg, "region_identity_spearman.csv")
        ctx.save_figure(plotting.heatmap(reg.astype(float), "Organoid cluster vs reference region (Spearman)"),
                        "region_identity.png")
        out["best_region_per_cluster"] = reg.astype(float).idxmax(axis=1).to_dict()
    out["neurodevelopment"] = {k: v for k, v in neurodevelopment_stages(q, seed=ctx.seed).items() if k != "coverage"}
    ctx.save_table(fid.stress_scores(q, seed=ctx.seed), "stress_panel_coverage.csv", index=False)
    cols = [c for c in q.obs.columns if c.startswith(("ref_", "fidelity", "devstage_", "maturation", "neurodev", "stress_"))]
    ctx.save_table(q.obs[cols], "organoid_cell_scores.csv")
    out["h5ad"] = _save(ctx, q, save_h5ad)
    return out


# ----------------------------------------------------------------------------- electrophysiology
def spikes_skill(ctx, input, max_units=None, refractory_ms=1.5):
    from .ephys.spikes import load_nwb_units, population_metrics
    trains, meta = load_nwb_units(input, max_units=max_units)
    m = population_metrics(trains, refractory_ms=refractory_ms)
    ctx.save_table(m, "unit_metrics.csv")
    if len(meta):
        ctx.save_table(meta, "unit_metadata.csv")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].hist(np.log10(m["firing_rate_hz"].clip(lower=1e-3)), bins=40, color=plotting.OKABE_ITO[4])
    axes[0].set_xlabel("log10 firing rate (Hz)")
    axes[1].hist(m["isi_cv"].dropna(), bins=40, color=plotting.OKABE_ITO[2]); axes[1].set_xlabel("ISI CV")
    axes[2].hist(m["refractory_violation_fraction"].dropna(), bins=40, color=plotting.OKABE_ITO[5])
    axes[2].set_xlabel("Refractory violation fraction")
    fig.tight_layout()
    ctx.save_figure(fig, "unit_metrics.png")
    return {"n_units": int(len(m)), "median_rate_hz": float(m["firing_rate_hz"].median()),
            "median_isi_cv": float(m["isi_cv"].median()),
            "fraction_units_refractory_violation_gt_1pct": float((m["refractory_violation_fraction"] > 0.01).mean())}


def eeg_bandpower_skill(ctx, input, picks="eeg"):
    import mne
    from .ephys.lfp_eeg import band_power
    raw = mne.io.read_raw(input, preload=True, verbose="ERROR")
    raw.pick(picks)
    bp = band_power(raw.get_data(), raw.info["sfreq"], channel_names=raw.ch_names)
    ctx.save_table(bp, "band_power.csv")
    rel = bp[[c for c in bp.columns if c.endswith("_rel")]]
    ctx.save_figure(plotting.heatmap(rel, "Relative band power per channel", cmap="viridis", center=None),
                    "band_power.png")
    return {"n_channels": int(len(bp)), "sfreq": float(raw.info["sfreq"]), "mean_relative_power": rel.mean().to_dict()}


# ----------------------------------------------------------------------------- modelling
def hh_skill(ctx, currents="0,2,4,6,7,8,10,15,20,30", g_na_scale=1.0, g_k_scale=1.0, T=500.0):
    from .modeling.neurons import HHParams, hh_fi_curve, simulate_hh
    I = [float(x) for x in str(currents).split(",")]
    p = HHParams(g_na_scale=g_na_scale, g_k_scale=g_k_scale)
    rates = hh_fi_curve(I, T=T, params=p)
    tr = simulate_hh(10.0, T=100.0, params=p)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot(I, rates, "o-", color=plotting.OKABE_ITO[4]); ax[0].set_xlabel("I (µA/cm²)"); ax[0].set_ylabel("Rate (Hz)")
    ax[0].set_title("Hodgkin–Huxley f–I curve")
    ax[1].plot(tr["t"], tr["V"], color="k", lw=0.8); ax[1].set_xlabel("t (ms)"); ax[1].set_ylabel("V (mV)")
    ax[1].set_title("I = 10 µA/cm²")
    fig.tight_layout()
    ctx.save_figure(fig, "hh_fi_curve.png")
    pd.DataFrame({"current": I, "rate_hz": rates}).to_csv(ctx.path("fi_curve.csv"), index=False)
    return {"rates_hz": dict(zip(map(str, I), rates.round(3).tolist())), "params": tr["params"]}


def network_skill(ctx, N_E=2000, g=5.0, eta=2.0, T=1000.0, scale_inhibition=1.0, scale_excitation=1.0):
    from .modeling.network import brunel_network
    r = brunel_network(N_E=N_E, g=g, eta=eta, T=T, scale_inhibition=scale_inhibition,
                       scale_excitation=scale_excitation, seed=ctx.seed)
    pop = r.pop("population_rate_hz")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(np.arange(len(pop)), pop, lw=0.6, color="k"); ax.set_xlabel("t (ms, after 100 ms burn-in)")
    ax.set_ylabel("Population rate (Hz)")
    fig.tight_layout()
    ctx.save_figure(fig, "population_rate.png")
    return r


# ----------------------------------------------------------------------------- morphology / imaging
def morphology_skill(ctx, input, sholl_step=10.0):
    from pathlib import Path as _P
    from .morphology.swc import morphometrics, read_swc, sholl
    paths = sorted(_P(input).glob("*.swc")) if _P(input).is_dir() else [_P(input)]
    rows, curves = [], []
    for p in paths:
        df = read_swc(p)
        rows.append({"file": p.name, **morphometrics(df)})
        s = sholl(df, step=sholl_step)
        s["file"] = p.name
        curves.append(s)
    m = pd.DataFrame(rows)
    ctx.save_table(m, "morphometrics.csv", index=False)
    sh = pd.concat(curves)
    ctx.save_table(sh, "sholl.csv", index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    for f, g in sh.groupby("file"):
        ax.plot(g["radius"], g["intersections"], lw=0.8, alpha=0.7)
    ax.set_xlabel("Radius (µm)"); ax.set_ylabel("Intersections"); ax.set_title("Sholl analysis")
    fig.tight_layout()
    ctx.save_figure(fig, "sholl.png")
    return {"n_neurons": int(len(m)), "median_total_length": float(m["total_length"].median()),
            "median_branch_points": float(m["n_branch_points"].median())}


def connectome_skill(ctx, timeseries_npz, kind="tangent", label_key="labels", density=0.15):
    from .imaging.connectome import classify, connectivity, graph_metrics, vectorize
    ctx.provenance.add_input(timeseries_npz, label="timeseries")
    z = np.load(timeseries_npz, allow_pickle=True)
    ts = list(z["timeseries"])
    mats = connectivity(ts, kind=kind)
    gm = pd.DataFrame([graph_metrics(m, density=density) for m in connectivity(ts, kind="correlation")])
    ctx.save_table(gm, "graph_metrics.csv", index=False)
    out = {"n_subjects": len(ts), "n_regions": int(mats.shape[1]), "graph_metrics_mean": gm.mean().to_dict()}
    if label_key in z.files:
        out["classification"] = classify(vectorize(mats), z[label_key], seed=ctx.seed)
    return out
