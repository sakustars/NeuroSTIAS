"""SWC neuron morphology: parsing, morphometrics and Sholl analysis.

SWC columns: id, type, x, y, z, radius, parent (parent = -1 for the root).
Types: 1 soma, 2 axon, 3 basal dendrite, 4 apical dendrite (others = custom).
Definitions follow NeuroM conventions where possible so experiment E10 can
check agreement: total length = sum of parent–child segment lengths (excluding
soma–soma segments); branch points = non-soma nodes with ≥ 2 children; tips =
non-soma nodes with 0 children; Sholl intersections = number of segments crossing
each sphere centred on the soma.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TYPE_NAMES = {1: "soma", 2: "axon", 3: "basal_dendrite", 4: "apical_dendrite"}


def read_swc(path: str | Path) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        rows.append([int(float(parts[0])), int(float(parts[1])), *map(float, parts[2:6]), int(float(parts[6]))])
    df = pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"]).set_index("id")
    if (df["parent"] == -1).sum() < 1:
        raise ValueError("SWC has no root node (parent = -1)")
    return df


def soma_center(df: pd.DataFrame) -> np.ndarray:
    soma = df[df["type"] == 1]
    src = soma if len(soma) else df[df["parent"] == -1]
    return src[["x", "y", "z"]].to_numpy().mean(axis=0)


def segments(df: pd.DataFrame) -> pd.DataFrame:
    ch = df[df["parent"] != -1]
    par = df.loc[ch["parent"]]
    seg = pd.DataFrame({"child": ch.index, "parent": ch["parent"].to_numpy(), "type": ch["type"].to_numpy(),
                        "parent_type": par["type"].to_numpy()})
    p0 = par[["x", "y", "z"]].to_numpy()
    p1 = ch[["x", "y", "z"]].to_numpy()
    seg[["x0", "y0", "z0"]] = p0
    seg[["x1", "y1", "z1"]] = p1
    seg["length"] = np.linalg.norm(p1 - p0, axis=1)
    return seg


def morphometrics(df: pd.DataFrame) -> dict:
    seg = segments(df)
    neurite = seg[seg["type"] != 1]
    children = df["parent"].value_counts()
    nonsoma = df[df["type"] != 1]
    n_children = nonsoma.index.map(lambda i: children.get(i, 0)).to_numpy()
    c = soma_center(df)
    xyz = nonsoma[["x", "y", "z"]].to_numpy()
    out = {
        "n_nodes": int(len(df)),
        "total_length": float(neurite["length"].sum()),
        "n_branch_points": int(np.sum(n_children >= 2)),
        "n_tips": int(np.sum(n_children == 0)),
        "max_radial_distance": float(np.linalg.norm(xyz - c, axis=1).max()) if len(xyz) else 0.0,
        "n_stems": int(((seg["parent_type"] == 1) & (seg["type"] != 1)).sum()),
    }
    for t, name in TYPE_NAMES.items():
        if t == 1:
            continue
        s = seg[seg["type"] == t]
        out[f"{name}_length"] = float(s["length"].sum())
    out["path_length_max"] = float(_max_path(df, seg))
    return out


def _max_path(df, seg) -> float:
    plen = {}
    order = list(df.index)
    lengths = dict(zip(seg["child"], seg["length"]))
    parent = df["parent"].to_dict()
    best = 0.0
    for nid in order:  # SWC parents precede children in valid files; fall back to recursion otherwise
        p = parent[nid]
        if p == -1:
            plen[nid] = 0.0
        else:
            if p not in plen:
                stack, cur = [], nid
                while parent[cur] != -1 and parent[cur] not in plen:
                    stack.append(parent[cur])
                    cur = parent[cur]
                for s in reversed(stack):
                    pp = parent[s]
                    plen[s] = (plen.get(pp, 0.0) if pp != -1 else 0.0) + lengths.get(s, 0.0)
            plen[nid] = plen[p] + (lengths[nid] if df.loc[nid, "type"] != 1 else 0.0)
        best = max(best, plen[nid])
    return best


def sholl(df: pd.DataFrame, step: float = 10.0, max_radius: float | None = None,
          neurite_types: tuple[int, ...] = (2, 3, 4)) -> pd.DataFrame:
    seg = segments(df)
    seg = seg[seg["type"].isin(neurite_types)]
    c = soma_center(df)
    d0 = np.linalg.norm(seg[["x0", "y0", "z0"]].to_numpy() - c, axis=1)
    d1 = np.linalg.norm(seg[["x1", "y1", "z1"]].to_numpy() - c, axis=1)
    lo, hi = np.minimum(d0, d1), np.maximum(d0, d1)
    R = np.arange(step, (max_radius or hi.max()) + step, step)
    counts = [int(np.sum((lo < r) & (hi >= r))) for r in R]
    return pd.DataFrame({"radius": R, "intersections": counts})
