"""Unit tests for Phase-2 modules on small synthetic inputs (tests only; never used as results)."""
import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse

from neurostias.atlas.annotate import annotate, score_panels
from neurostias.core import graph
from neurostias.spatial3d.communication import _abundance, communication
from neurostias.spatial3d.domains import DomainParams, detect_domains
from neurostias.spatial3d.reconstruct import apply_affine, register_rigid


def _toy(n_per=150, seed=0, two_blocks=True):
    """Two spatial halves with different marker programmes, on a grid, two sections per block."""
    rng = np.random.default_rng(seed)
    genes = ["SLC17A7", "SATB2", "NEUROD6", "CAMK2A", "GFAP", "AQP4", "SLC1A2", "GJA1",
             "MBP", "PLP1", "MOBP", "MOG"] + [f"G{i}" for i in range(300)]
    rows, obs, xyz = [], [], []
    blocks = ["m1", "m2"] if two_blocks else ["m1"]
    for b in blocks:
        for s, z in ((0, 0.0), (1, 20.0)):
            g = int(np.sqrt(n_per))
            xs, ys = np.meshgrid(np.arange(g), np.arange(g))
            for x, y in zip(xs.ravel(), ys.ravel()):
                lam = rng.gamma(1.0, 0.3, len(genes))
                if x < g / 2:
                    lam[0:4] += 4  # excitatory neuron side
                else:
                    lam[8:12] += 4  # oligodendrocyte side
                rows.append(rng.poisson(lam))
                obs.append({"sample_id": b, "section": f"{b}_s{s}", "truth": "neuron" if x < g / 2 else "oligo"})
                xyz.append([x * 100.0, y * 100.0, z])
    a = ad.AnnData(X=sparse.csr_matrix(np.array(rows, dtype=float)), obs=pd.DataFrame(obs))
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.var_names = genes
    a.layers["log_normalized"] = sparse.csr_matrix(np.log1p(a.X.toarray()))
    a.obsm["spatial_3d"] = np.array(xyz)
    a.obsm["spatial"] = a.obsm["spatial_3d"][:, :2]
    return a


def test_knn_graph_never_crosses_blocks():
    a = _toy()
    blocks, _ = graph.block_labels(a.obs)
    W = graph.knn_graph(a.obsm["spatial_3d"], blocks, k=6).tocoo()
    assert (blocks[W.row] == blocks[W.col]).all()


def test_z_scale_zero_is_per_section():
    a = _toy()
    blocks, _ = graph.block_labels(a.obs)
    sec = a.obs["section"].to_numpy()
    W = graph.knn_graph(a.obsm["spatial_3d"], blocks, k=6, z_scale=0.0, sections=sec).tocoo()
    assert (sec[W.row] == sec[W.col]).all()
    W3 = graph.cross_section_knn(a.obsm["spatial_3d"], sec, blocks, k_within=6, k_between=3).tocoo()
    assert (sec[W3.row] != sec[W3.col]).any() and (blocks[W3.row] == blocks[W3.col]).all()


def test_annotation_recovers_classes_and_leaves_missing_panels_unusable():
    a = _toy()
    res = annotate(a, species="human", n_null=300)
    calls = a.obs["neuro_class"].astype(str)
    assert (calls[a.obs.truth == "neuron"] == "Excitatory neuron").mean() > 0.8
    assert (calls[a.obs.truth == "oligo"] == "Oligodendrocyte").mean() > 0.8
    assert "Microglia" in res["summary"]["unusable_class_panels"]


def test_annotation_unresolved_without_signal():
    rng = np.random.default_rng(1)
    genes = ["SLC17A7", "SATB2", "NEUROD6", "MBP", "PLP1", "MOBP"] + [f"G{i}" for i in range(200)]
    X = rng.poisson(0.5, (300, len(genes))).astype(float)
    a = ad.AnnData(X=sparse.csr_matrix(X))
    a.var_names = genes
    a.layers["log_normalized"] = sparse.csr_matrix(np.log1p(X))
    annotate(a, species="human", n_null=300)
    assert (a.obs["neuro_class"] == "Unresolved").mean() > 0.85


def test_rigid_registration_recovers_known_transform():
    rng = np.random.default_rng(0)
    target = rng.uniform(0, 3000, (500, 2))
    theta = np.deg2rad(7.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    source = (target - 1500) @ R.T + 1500 + np.array([120.0, -80.0])
    aff, info = register_rigid(source, target, overlap_threshold=50)
    assert np.abs(apply_affine(source, aff) - target).max() < 1e-6
    assert abs(info["rotation_degrees"] + 7.0) < 1e-6


def test_domains_split_the_two_halves():
    a = _toy()
    detect_domains(a, DomainParams(n_domains=2, n_pcs=10, n_hvg=200), section_key="section", block_key="sample_id")
    ct = pd.crosstab(a.obs["neuro_domain"], a.obs["truth"])
    purity = ct.max(axis=1).sum() / ct.values.sum()
    assert purity > 0.9


def test_transmitter_abundance_requires_all_groups():
    X = sparse.csr_matrix(np.array([[1.0, 0.0, 2.0], [1.0, 1.0, 0.0]]))
    lk = {"GAD1": 0, "GAD2": 1, "SLC32A1": 2}
    v = _abundance("GAD1|GAD2;SLC32A1", X, lk, "human")
    assert v[0] > 0 and v[1] == 0  # second row lacks VGAT


def test_communication_runs_and_reports_replicates():
    a = _toy()
    a.obs["label"] = a.obs["truth"]
    res = communication(a, "label", database="generic_v441", n_permutations=20, block_key="sample_id")
    assert isinstance(res, pd.DataFrame)
