"""E06 - Neurotransmitter-aware, replicate-consistent spatial communication.

Ground truth for cell-cell communication is weak, so two pre-registered checks are used.

Check 1 - recall of canonical interactions. A list of textbook class-level interactions is
fixed in this file before running (CANONICAL). For each method, an interaction counts as
recovered if it is reported significant for the stated sender -> receiver classes.
Datasets: MERFISH hypothalamus (author classes) and DLPFC (NeuroSTIAS classes).
Methods: NeuroSTIAS neuro database (spatial, within-sample permutation), the 3D-STIAS V4.4.1
generic database with the same statistics, and LIANA consensus (non-spatial; top-ranked
interactions with magnitude_rank < 0.05 counted as significant).

Check 2 - cross-donor replication (DLPFC, 3 donors). Interactions are discovered on the
sections of one donor and tested on each other donor; replication rate = fraction of
discoveries also significant in the held-out donor. Compared: NeuroSTIAS with replicate
filtering (sections as replicates), NeuroSTIAS pooled (no replicate filter), and LIANA.
"""
import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neurostias import data  # noqa: E402
from neurostias.atlas.annotate import annotate  # noqa: E402
from neurostias.experiment import Experiment, md_table, run_baseline_script  # noqa: E402
from neurostias.spatial3d.communication import communication  # noqa: E402

N = ["Excitatory neuron", "Inhibitory neuron"]
CANONICAL = [  # (interaction id in neuro DB, generic DB ligand_receptor or None, senders, receivers, reference)
    ("PDGFB_PDGFRB", "PDGFB_PDGFRB", ["Endothelial"], ["Pericyte"], "Armulik et al. 2010 Nature"),
    ("CX3CL1_CX3CR1", None, N, ["Microglia"], "Paolicelli et al. 2011 Science"),
    ("IL34_CSF1R", None, N, ["Microglia"], "Greter et al. 2012 Immunity"),
    ("CSF1_CSF1R", "CSF1_CSF1R", ["Astrocyte", "Oligodendrocyte", "OPC"] + N, ["Microglia"], "Elmore et al. 2014 Neuron"),
    ("GABA_A", None, ["Inhibitory neuron"], N, "textbook (fast inhibition)"),
    ("GLU_AMPA", None, ["Excitatory neuron"], N, "textbook (fast excitation)"),
    ("GLU_NMDA_2B", None, ["Excitatory neuron"], N, "textbook"),
    ("VEGFA_R", "VEGFA_KDR", ["Astrocyte"], ["Endothelial"], "Argaw et al. 2012 J Clin Invest"),
    ("PTN_PTPRZ1", None, N + ["Astrocyte"], ["OPC", "Astrocyte"], "Kuboyama et al. 2015"),
]
MERFISH_MAP = {"Excitatory": "Excitatory neuron", "Inhibitory": "Inhibitory neuron", "Astrocyte": "Astrocyte",
               "OD Mature": "Oligodendrocyte", "OD Immature": "OPC", "Microglia": "Microglia",
               "Endothelial": "Endothelial", "Pericytes": "Pericyte", "Ependymal": "Ependymal"}


def recovered(res, db):
    out = {}
    for iid, gen, snd, rcv, _ in CANONICAL:
        key = iid if db == "neuro" else gen
        if key is None:
            out[iid] = "not_in_database"
            continue
        sub = res[(res["interaction"] == key) & res["sender"].isin(snd) & res["receiver"].isin(rcv)] if len(res) else res
        if len(sub) == 0:
            out[iid] = "not_tested"
        else:
            out[iid] = "recovered" if sub["significant"].any() else "not_significant"
    return out


def liana_recovered(lres, species):
    """LIANA: complex names joined by '_' ; check ligand/receptor gene membership."""
    pairs = {"PDGFB_PDGFRB": ("PDGFB", "PDGFRB"), "CX3CL1_CX3CR1": ("CX3CL1", "CX3CR1"), "IL34_CSF1R": ("IL34", "CSF1R"),
             "CSF1_CSF1R": ("CSF1", "CSF1R"), "VEGFA_R": ("VEGFA", "KDR"), "PTN_PTPRZ1": ("PTN", "PTPRZ1"),
             "GABA_A": (None, None), "GLU_AMPA": (None, None), "GLU_NMDA_2B": (None, None)}
    out = {}
    for iid, _, snd, rcv, _ in CANONICAL:
        lg, rc = pairs[iid]
        if lg is None:
            out[iid] = "not_in_database"
            continue
        if species == "mouse":
            lg, rc = lg.capitalize(), rc.capitalize()
        sub = lres[(lres["ligand_complex"].str.upper() == lg.upper()) & lres["receptor_complex"].str.upper().str.contains(rc.upper())
                   & lres["source"].isin(snd) & lres["target"].isin(rcv)]
        if len(sub) == 0:
            out[iid] = "not_tested"
        else:
            out[iid] = "recovered" if (sub["magnitude_rank"] < 0.05).any() else "not_significant"
    return out


def run_liana(adata, label, species):
    with tempfile.TemporaryDirectory() as td:
        ip, op = Path(td) / "x.h5ad", Path(td) / "o.csv"
        b = sc.AnnData(X=adata.layers["log_normalized"], obs=pd.DataFrame({label: adata.obs[label].astype(str).to_numpy()},
                                                                         index=adata.obs_names))
        b.var_names = adata.var_names
        b.write_h5ad(ip)
        ok, log = run_baseline_script(ROOT / "benchmarks/baselines/liana_rank.py",
                                      ["--input", str(ip), "--groupby", label, "--out", str(op)] + (["--mouse"] if species == "mouse" else []),
                                      timeout=7200)
        return (pd.read_csv(op) if ok else None), log


def merfish(exp):
    p = data.path("merfish_hypothalamus")
    exp.prov.add_input(p, "merfish")
    a = sc.read_h5ad(p)
    a.obs["cls"] = a.obs["Cell_class"].astype(str).str.replace(r" \d+$", "", regex=True).map(MERFISH_MAP)
    a = a[a.obs["cls"].notna().to_numpy()].copy()
    X = a.X.tocsr()
    lib = np.asarray(X.sum(1)).ravel()
    a.layers["log_normalized"] = sparse.csr_matrix(np.log1p(X.multiply(np.median(lib) / np.maximum(lib, 1e-9)[:, None]).toarray()))
    a.obsm["spatial_3d"] = np.column_stack([a.obs[["Centroid_X", "Centroid_Y"]].to_numpy(float), a.obs["Bregma"].astype(float) * 10])
    a.obs["section"] = a.obs["Bregma"].astype(str)
    out = {}
    for db in ("neuro", "generic_v441"):
        r = communication(a, "cls", database=db, k=10, z_scale=0.0, section_key="section", n_permutations=200, species="mouse")
        exp.table(r, f"e06_merfish_{db}.csv")
        out[f"neurostias_{db}"] = recovered(r, "neuro" if db == "neuro" else "generic")
        out[f"neurostias_{db}_n_significant"] = int(r["significant"].sum()) if len(r) else 0
    l, log = run_liana(a, "cls", "mouse")
    if l is not None:
        exp.table(l, "e06_merfish_liana.csv")
        out["liana"] = liana_recovered(l, "mouse")
        exp.baseline("liana", "ran")
    else:
        exp.baseline("liana", "failed", log)
    return out


def dlpfc_load():
    parts = []
    for s in ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675", "151676"]:
        x = sc.read_h5ad(data.path("dlpfc", f"{s}.h5ad"))
        x.obs_names = [f"{s}_{n}" for n in x.obs_names]
        parts.append(x)
    a = sc.concat(parts, join="inner")
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.layers["log_normalized"] = a.X.copy()
    annotate(a, species="human", seed=0)
    return a


def dlpfc(exp):
    a = dlpfc_load()
    exp.prov.add_input(data.path("dlpfc"), "dlpfc_all")
    exp.table(a.obs.groupby(["donor", "neuro_class"], observed=True).size().unstack(fill_value=0), "e06_dlpfc_class_counts.csv", index=True)
    lab = "neuro_class"
    a = a[a.obs[lab].astype(str) != "Unresolved"].copy()
    rec = {}
    for db in ("neuro", "generic_v441"):
        r = communication(a, lab, database=db, k=6, z_scale=0.0, block_key="section", section_key="section",
                          n_permutations=200, species="human")
        exp.table(r, f"e06_dlpfc_{db}_all.csv")
        rec[f"neurostias_{db}"] = recovered(r, "neuro" if db == "neuro" else "generic")
    l, log = run_liana(a, lab, "human")
    if l is not None:
        rec["liana"] = liana_recovered(l, "human")
    # cross-donor replication
    reps = []
    donors = a.obs["donor"].unique()
    for d in donors:
        disc = a[a.obs["donor"] == d].copy()
        res_d = communication(disc, lab, database="neuro", k=6, z_scale=0.0, block_key="section", section_key="section",
                              n_permutations=200, species="human", min_replicate_fraction=0.75)
        pooled = res_d.assign(significant=res_d["padj"] < 0.05)
        ld, _ = run_liana(disc, lab, "human")
        for o in donors:
            if o == d:
                continue
            val = a[a.obs["donor"] == o].copy()
            res_o = communication(val, lab, database="neuro", k=6, z_scale=0.0, block_key="section", section_key="section",
                                  n_permutations=200, species="human", min_replicate_fraction=0.75)
            key = ["interaction", "sender", "receiver"]
            sig_o = set(map(tuple, res_o.loc[res_o["padj"] < 0.05, key].to_numpy()))
            for name, rd in (("neurostias_replicate_filtered", res_d), ("neurostias_pooled", pooled)):
                disc_set = set(map(tuple, rd.loc[rd["significant"], key].to_numpy()))
                reps.append({"discovery_donor": d, "validation_donor": o, "method": name, "n_discoveries": len(disc_set),
                             "replication_rate": len(disc_set & sig_o) / max(len(disc_set), 1)})
            if ld is not None:
                lo, _ = run_liana(val, lab, "human")
                k2 = ["ligand_complex", "receptor_complex", "source", "target"]
                ds = set(map(tuple, ld.loc[ld["magnitude_rank"] < 0.05, k2].to_numpy()))
                vs = set(map(tuple, lo.loc[lo["magnitude_rank"] < 0.05, k2].to_numpy())) if lo is not None else set()
                reps.append({"discovery_donor": d, "validation_donor": o, "method": "liana", "n_discoveries": len(ds),
                             "replication_rate": len(ds & vs) / max(len(ds), 1)})
        print("donor", d, "done", flush=True)
    return rec, pd.DataFrame(reps)


def main(quick):
    exp = Experiment("E06", "Neurotransmitter-aware spatial communication", "Does a neurotransmitter/neuropeptide-aware, "
                     "replicate-consistent spatial method recover canonical neural interactions and replicate across "
                     "donors better than a generic database and a non-spatial consensus method?",
                     params={"n_permutations": 200, "k_merfish": 10, "k_dlpfc": 6, "canonical_list": [c[:4] for c in CANONICAL]})
    m = merfish(exp)
    rec, reps = dlpfc(exp)
    exp.table(reps, "e06_dlpfc_cross_donor_replication.csv")
    rep_sum = reps.groupby("method", as_index=False).agg(mean_replication_rate=("replication_rate", "mean"),
                                                         mean_n_discoveries=("n_discoveries", "mean"))
    exp.table(rep_sum, "e06_replication_summary.csv")
    rows = []
    for ds, dct in (("MERFISH", m), ("DLPFC", rec)):
        for meth, r in dct.items():
            if not isinstance(r, dict):
                continue
            for iid, status in r.items():
                rows.append({"dataset": ds, "method": meth, "interaction": iid, "status": status})
    recall = pd.DataFrame(rows)
    exp.table(recall, "e06_canonical_recall_long.csv")
    piv = recall.assign(hit=recall.status.eq("recovered")).groupby(["dataset", "method"], as_index=False).agg(
        n_recovered=("hit", "sum"), n_canonical=("hit", "size"),
        n_not_in_db=("status", lambda s: int((s == "not_in_database").sum())))
    exp.table(piv, "e06_canonical_recall_summary.csv")
    exp.summary.update({"canonical_recall": piv.to_dict("records"), "replication": rep_sum.to_dict("records"),
                        "merfish_n_significant": {k: v for k, v in m.items() if k.endswith("n_significant")}})
    exp.note("The canonical list was fixed in run.py before any results were inspected; it is small and textbook-level, "
             "so recall is a sanity check rather than a comprehensive benchmark.")
    exp.note("LIANA is non-spatial (cluster-level) and uses its own consensus resource; its significance threshold here "
             "(magnitude_rank < 0.05) follows the LIANA documentation's suggested cut-off.")
    exp.note("DLPFC classes come from NeuroSTIAS annotation of Visium spots (mixtures), so receiver/sender classes are "
             "dominant-class approximations.")
    exp.finish([("Canonical interaction recall", md_table(piv)),
                ("Canonical recall detail", md_table(recall.pivot_table(index=["dataset", "interaction"], columns="method",
                                                                         values="status", aggfunc="first").reset_index())),
                ("Cross-donor replication (DLPFC)", md_table(rep_sum))])
    print(piv.to_string()); print(rep_sum.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a, _ = ap.parse_known_args()
    main(a.quick)
