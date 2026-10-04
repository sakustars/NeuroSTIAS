"""Serial-section registration and 3D reconstruction.

Rigid registration uses multi-start trimmed ICP without reflection and with
deterministic tie-breaks (ported from the GSE214349 importer,
``prepare_gse214349.py``). Sections are aligned sequentially to their predecessor
within each biological sample; sections from different samples are never aligned
to each other. ``z`` comes from an explicit per-section position (µm) when known,
otherwise from section order × ``z_spacing``, and the source is recorded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def _rotation(angle_degrees: float) -> np.ndarray:
    r = np.deg2rad(angle_degrees)
    return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation + translation (Kabsch), reflection forbidden."""
    sc, tc = source.mean(axis=0), target.mean(axis=0)
    U, _, Vt = np.linalg.svd((source - sc).T @ (target - tc), full_matrices=False)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, tc - R @ sc


def overlap_score(source: np.ndarray, target: np.ndarray, threshold: float) -> float:
    a = cKDTree(target).query(source, k=1)[0]
    b = cKDTree(source).query(target, k=1)[0]
    return float(0.5 * (np.mean(a <= threshold) + np.mean(b <= threshold)))


def register_rigid(source: np.ndarray, target: np.ndarray, angle_range: float = 15.0, angle_step: float = 1.0,
                   trim: float = 0.80, max_iter: int = 80, overlap_threshold: float = 125.0,
                   global_search: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Register 2D point cloud ``source`` onto ``target``. Returns (3x3 affine, audit dict).

    Default: multi-start ICP from initial rotations within +/-``angle_range`` degrees (serial
    sections are usually mounted with similar orientation). ``global_search=True`` first scans
    the full circle in 10-degree steps and then refines around the best start in 1-degree steps,
    for sections with arbitrary orientation.
    """
    if global_search:
        coarse, _ = register_rigid(source, target, angle_range=180.0, angle_step=10.0, trim=trim, max_iter=40,
                                   overlap_threshold=overlap_threshold)
        a0 = float(np.rad2deg(np.arctan2(coarse[1, 0], coarse[0, 0])))
        best = None
        for a in np.arange(a0 - 10, a0 + 10 + 1e-9, 1.0):
            aff, info = _register_from(source, target, [a], trim, max_iter, overlap_threshold)
            if best is None or (info["overlap_after"], -info["trimmed_rmse"]) > (best[1]["overlap_after"], -best[1]["trimmed_rmse"]):
                best = (aff, info)
        best[1]["global_search"] = True
        return best
    return _register_from(source, target, list(np.arange(-angle_range, angle_range + 1e-9, angle_step)), trim, max_iter,
                          overlap_threshold)


def _register_from(source, target, starts, trim, max_iter, overlap_threshold):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if min(len(source), len(target)) < 20:
        raise ValueError("a section needs at least 20 observations for registration")
    s_c, t_c = source.mean(axis=0), target.mean(axis=0)
    tree = cKDTree(target)
    before = overlap_score(source - s_c + t_c, target, overlap_threshold)
    best = None
    for a0 in starts:
        R = _rotation(float(a0))
        t = t_c - R @ s_c
        prev = np.inf
        it = 0
        for it in range(1, max_iter + 1):
            moved = source @ R.T + t
            d, nn = tree.query(moved, k=1)
            keep = d <= max(float(np.quantile(d, trim)), 1e-9)
            dR, dt = rigid_fit(moved[keep], target[nn[keep]])
            R, t = dR @ R, dR @ t + dt
            obj = float(np.mean(d[keep] ** 2))
            if np.isfinite(prev) and abs(prev - obj) <= 1e-7 * max(1.0, prev):
                break
            prev = obj
        score = overlap_score(source @ R.T + t, target, overlap_threshold)
        cand = (round(score, 12), round(-prev, 6), R, t, it)
        if best is None or cand[:2] > best[:2]:
            best = cand
    score, neg_obj, R, t, it = best
    affine = np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1.0]])
    return affine, {
        "overlap_before": before, "overlap_after": float(score),
        "trimmed_rmse": float(np.sqrt(max(0.0, -neg_obj))),
        "rotation_degrees": float(np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))),
        "translation": [float(t[0]), float(t[1])], "iterations": int(it),
    }


def apply_affine(xy: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return xy @ affine[:2, :2].T + affine[:2, 2]


def reconstruct(adata, section_key: str, block_key: str | None = None, coord_key: str = "spatial",
                z_key: str | None = None, z_spacing: float | None = None, order_key: str | None = None,
                register: bool = True, overlap_threshold: float | None = None) -> pd.DataFrame:
    """Build ``obsm['spatial_3d']`` from per-section 2D coordinates. Returns an audit table.

    z source priority: ``z_key`` (per-observation µm) > ``order_key`` × ``z_spacing``
    > section order (sorted by ``order_key`` or name) × ``z_spacing``.
    """
    xy_all = np.asarray(adata.obsm[coord_key], dtype=float)[:, :2]
    sections = adata.obs[section_key].astype(str).to_numpy()
    blocks = adata.obs[block_key].astype(str).to_numpy() if block_key else np.repeat("all", adata.n_obs)
    out = np.zeros((adata.n_obs, 3))
    rows = []
    for b in pd.unique(blocks):
        bm = blocks == b
        secs = pd.unique(sections[bm])
        if order_key:
            order_val = adata.obs.loc[bm, [section_key, order_key]].drop_duplicates().set_index(section_key)[order_key]
            secs = sorted(secs, key=lambda s: float(order_val[s]))
        else:
            secs = sorted(secs)
        if overlap_threshold is None:
            first = xy_all[bm & (sections == secs[0])]
            spacing = np.median(cKDTree(first).query(first, k=2)[0][:, 1]) if len(first) > 1 else 1.0
            thr = 1.25 * spacing
        else:
            thr = overlap_threshold
        prev_xy = None
        for i, s in enumerate(secs):
            m = bm & (sections == s)
            xy = xy_all[m]
            audit: dict[str, Any] = {"block": b, "section": s, "order": i, "n_obs": int(m.sum())}
            if register and prev_xy is not None:
                aff, info = register_rigid(xy, prev_xy, overlap_threshold=thr)
                xy = apply_affine(xy, aff)
                audit.update(info)
                audit["affine"] = aff.tolist()
            else:
                audit["affine"] = np.eye(3).tolist()
            if z_key:
                z = adata.obs.loc[m, z_key].astype(float).to_numpy()
                audit["z_source"] = z_key
            elif z_spacing is not None:
                base = float(adata.obs.loc[m, order_key].astype(float).iloc[0]) if order_key else i
                z = np.full(m.sum(), base * z_spacing)
                audit["z_source"] = f"{order_key or 'order'} x {z_spacing}"
            else:
                z = np.full(m.sum(), float(i))
                audit["z_source"] = "order index (unitless)"
            audit["z"] = float(np.median(z))
            out[m] = np.column_stack([xy, z])
            prev_xy = xy
            rows.append(audit)
    adata.obsm["spatial_3d"] = out
    audit_df = pd.DataFrame(rows)
    adata.uns["reconstruction"] = {"section_key": section_key, "block_key": block_key,
                                   "z_spacing": z_spacing, "z_key": z_key, "order_key": order_key}
    return audit_df
