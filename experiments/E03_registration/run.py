"""E03 - Serial-section registration accuracy (semi_synthetic, built from real sections).

For each real section, a "moved" copy is created: 10% of observations dropped, positions
jittered (sigma = 10 um), counts binomially thinned to 50% (so expression differs as between
real sections), then rotated by theta and translated by up to 500 um. The true position of
every retained observation is known, so registration error (um) is exact.
Rotation regimes: |theta| <= 15 deg (inside the NeuroSTIAS ICP search range) and
15 < |theta| <= 45 deg (outside it) to expose failure modes honestly.
Methods: NeuroSTIAS rigid ICP (geometry only; default +/-15 deg starts, and global_search=True
which scans the full circle - added after the quick test confirmed the documented +/-15 deg
limitation, and reported separately), PASTE (expression + space OT, alpha=0.1),
centroid alignment (translation only).
Error metric: after mapping the moved copy back, the mean distance between each observation's
recovered and true positions, computed in the target frame (PASTE's output frame is re-centred,
so its target positions are mapped too and errors are measured relative to them).
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

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neurostias import data  # noqa: E402
from neurostias.experiment import Experiment, md_table, run_baseline_script  # noqa: E402
from neurostias.spatial3d.reconstruct import apply_affine, register_rigid  # noqa: E402

UM_PER_PX = 100.0 / 137.0


def sections():
    out = []
    for s in ["151507", "151509", "151669", "151671", "151673", "151675"]:
        a = sc.read_h5ad(data.path("dlpfc", f"{s}.h5ad"))
        a.X = a.layers["counts"]
        a.obsm["spatial"] = a.obsm["spatial"] * UM_PER_PX
        out.append(("DLPFC", s, a))
    g = sc.read_h5ad(data.path("gse214349_3d"))
    for sl in ["IVH_1d_1_50", "IVH_3d_3_110", "IVH_7d_2_80", "IVH_C1_50", "IVH_S1_80"]:
        a = g[g.obs["slice_id"].astype(str) == sl].copy()
        a.obsm["spatial"] = np.asarray(a.obsm["spatial_local_um"])
        out.append(("GSE214349", sl, a))
    return out


def make_moved(a, theta_deg, shift, rng):
    keep = rng.random(a.n_obs) > 0.10
    src = a[keep].copy()
    X = src.X.tocsr().astype(np.int64).copy()
    X.data = rng.binomial(X.data, 0.5)
    X.eliminate_zeros()
    src.X = X.astype(np.float32)
    true_xy = np.asarray(src.obsm["spatial"], float)
    jit = true_xy + rng.normal(0, 10.0, true_xy.shape)
    c = jit.mean(0)
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    moved = (jit - c) @ R.T + c + shift
    src.obsm["spatial"] = moved
    return src, true_xy, keep


def main(quick):
    exp = Experiment("E03", "Serial-section registration accuracy", "How accurately do NeuroSTIAS ICP and PASTE recover "
                     "known rigid transforms between real sections, and where does each fail?",
                     data_origin="semi_synthetic", params={"drop": 0.10, "jitter_um": 10, "thinning": 0.5,
                                                            "max_shift_um": 500, "paste_alpha": 0.1})
    rng = np.random.default_rng(0)
    secs = sections()
    if quick:
        secs = secs[:1] + secs[6:7]
    rows = []
    for ds, sid, a in secs:
        for rep in range(1 if quick else 3):
            for regime, lo, hi in (("<=15deg", 0, 15), ("15-45deg", 15, 45)):
                theta = float(rng.uniform(lo, hi) * rng.choice([-1, 1]))
                shift = rng.uniform(-500, 500, 2)
                src, true_xy, keep = make_moved(a, theta, shift, rng)
                tgt_xy = np.asarray(a.obsm["spatial"], float)
                spacing = 100.0
                # NeuroSTIAS ICP
                t0 = time.time()
                aff, info = register_rigid(np.asarray(src.obsm["spatial"]), tgt_xy, overlap_threshold=1.25 * spacing)
                rec = apply_affine(np.asarray(src.obsm["spatial"]), aff)
                rows.append({"dataset": ds, "section": sid, "rep": rep, "regime": regime, "theta": theta,
                             "method": "neurostias_icp", "error_um": float(np.mean(np.linalg.norm(rec - true_xy, axis=1))),
                             "seconds": time.time() - t0})
                t0 = time.time()
                aff, info = register_rigid(np.asarray(src.obsm["spatial"]), tgt_xy, overlap_threshold=1.25 * spacing,
                                           global_search=True)
                rec = apply_affine(np.asarray(src.obsm["spatial"]), aff)
                rows.append({"dataset": ds, "section": sid, "rep": rep, "regime": regime, "theta": theta,
                             "method": "neurostias_icp_global", "error_um": float(np.mean(np.linalg.norm(rec - true_xy, axis=1))),
                             "seconds": time.time() - t0})
                # centroid only
                rec = np.asarray(src.obsm["spatial"]) - np.asarray(src.obsm["spatial"]).mean(0) + tgt_xy[keep].mean(0)
                rows.append({"dataset": ds, "section": sid, "rep": rep, "regime": regime, "theta": theta,
                             "method": "centroid_only", "error_um": float(np.mean(np.linalg.norm(rec - true_xy, axis=1))),
                             "seconds": 0.0})
                # PASTE
                with tempfile.TemporaryDirectory() as td:
                    tp, sp, op = Path(td) / "t.h5ad", Path(td) / "s.h5ad", Path(td) / "o.npy"
                    tt = a.copy(); tt.obs = tt.obs[[]]; tt.var = tt.var[[]]
                    ss = src.copy(); ss.obs = ss.obs[[]]; ss.var = ss.var[[]]
                    tt.write_h5ad(tp); ss.write_h5ad(sp)
                    t0 = time.time()
                    ok, log = run_baseline_script(ROOT / "benchmarks/baselines/paste_align.py",
                                                  ["--target", str(tp), "--source", str(sp), "--alpha", "0.1", "--out", str(op)],
                                                  timeout=3600, env="paste")
                    if ok:
                        ns = np.load(op)
                        nt = np.load(str(op).replace(".npy", "_target.npy"))
                        # PASTE re-centres both slices; map target back to original frame by its translation
                        offset = tgt_xy.mean(0) - nt.mean(0)
                        rec = ns + offset
                        rows.append({"dataset": ds, "section": sid, "rep": rep, "regime": regime, "theta": theta,
                                     "method": "paste", "error_um": float(np.mean(np.linalg.norm(rec - true_xy, axis=1))),
                                     "seconds": time.time() - t0})
                        exp.baseline("paste", "ran")
                    else:
                        exp.baseline("paste", "failed", log)
                print(ds, sid, rep, regime, flush=True)
    raw = pd.DataFrame(rows)
    exp.table(raw, "e03_registration_raw_semi_synthetic.csv")
    summ = raw.groupby(["regime", "method"], as_index=False).agg(
        median_error_um=("error_um", "median"), mean_error_um=("error_um", "mean"),
        frac_error_below_50um=("error_um", lambda x: float(np.mean(x < 50))), median_seconds=("seconds", "median"),
        n=("error_um", "size"))
    exp.table(summ, "e03_summary_semi_synthetic.csv")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for mth, mk in (("neurostias_icp", "o"), ("neurostias_icp_global", "D"), ("paste", "s"), ("centroid_only", "^")):
        d = raw[raw.method == mth]
        ax.scatter(d["theta"].abs(), d["error_um"], label=mth, marker=mk, s=18, alpha=0.8)
    ax.axvline(15, color="k", ls="--", lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("|rotation| (deg)"); ax.set_ylabel("Mean registration error (um)")
    ax.legend(frameon=False); ax.set_title("E03 (semi-synthetic from real sections)")
    exp.figure(fig, "e03_error_vs_rotation.png")
    exp.summary.update({"summary": summ.to_dict("records"), "quick_mode": quick})
    exp.note("All results are semi-synthetic (known transforms applied to real sections). Real adjacent-section "
             "registration is assessed separately in E02 Part A via manual-layer agreement.")
    exp.note("The NeuroSTIAS ICP searches initial rotations within +/-15 deg by default; the 15-45 deg regime "
             "deliberately tests outside that range. The global_search variant was added after a quick test showed the "
             "expected failure outside +/-15 deg; both are reported.")
    exp.note("Noise floor: positions were jittered with sigma = 10 um, so a perfect registration still shows a mean "
             "error of 10*sqrt(pi/2) = 12.53 um (mean 2D displacement of the jitter); errors near 12.5 um are therefore at the floor.")
    exp.note("PASTE runs in its own environment (POT 0.9.0, NumPy 1.26, SciPy 1.11) because paste-bio 1.4 is "
             "incompatible with POT 0.9.7.")
    exp.finish([("Summary", md_table(summ))])
    print(summ.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a, _ = ap.parse_known_args()
    main(a.quick)
