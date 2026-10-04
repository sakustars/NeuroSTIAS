"""Modality unit tests with analytically known answers."""
import numpy as np
import pytest

from neurostias.ephys.lfp_eeg import band_power
from neurostias.ephys.spikes import psth, unit_metrics
from neurostias.modeling.neurons import HHParams, hh_fi_curve, lif_fi_curve, lif_rate_analytic
from neurostias.morphology.swc import morphometrics, read_swc, sholl


def test_regular_and_poisson_trains():
    reg = np.arange(0, 100, 0.1)  # 10 Hz regular
    m = unit_metrics(reg)
    assert abs(m["firing_rate_hz"] - 10) < 0.02 and m["isi_cv"] < 1e-9 and m["lv"] < 1e-9
    rng = np.random.default_rng(0)
    poi = np.cumsum(rng.exponential(0.05, 20000))
    mp = unit_metrics(poi)
    assert abs(mp["isi_cv"] - 1) < 0.03 and abs(mp["lv"] - 1) < 0.05 and abs(mp["fano_factor"] - 1) < 0.15


def test_psth_counts():
    st = np.array([1.05, 2.05, 3.05])
    centers, rate = psth(st, np.array([1.0, 2.0, 3.0]), window=(0, 0.1), bin_s=0.1)
    assert rate[0] == pytest.approx(10.0)


def test_band_power_sinusoid_is_alpha():
    sf = 250.0
    t = np.arange(0, 60, 1 / sf)
    x = np.sin(2 * np.pi * 10 * t) + 0.01 * np.random.default_rng(0).standard_normal(len(t))
    bp = band_power(x[None, :], sf)
    assert bp["alpha_rel"].iloc[0] > 0.95


def test_lif_matches_closed_form():
    I = np.array([2.1, 3.0, 5.0])
    assert np.allclose(lif_fi_curve(I, T=4000), lif_rate_analytic(I), atol=0.5)


def test_hh_repetitive_firing_threshold_and_ttx():
    r = hh_fi_curve([5.0, 7.0], T=300)
    assert r[0] == 0 and r[1] > 40
    assert hh_fi_curve([10.0], T=200, params=HHParams(g_na_scale=0.0))[0] == 0


def test_swc_morphometrics_and_sholl(tmp_path):
    # soma at origin; one dendrite 0->(10,0,0)->(20,0,0) that branches to (30,10,0) and (30,-10,0)
    swc = """# toy
1 1 0 0 0 5 -1
2 3 10 0 0 1 1
3 3 20 0 0 1 2
4 3 30 10 0 1 3
5 3 30 -10 0 1 3
"""
    p = tmp_path / "toy.swc"
    p.write_text(swc)
    df = read_swc(p)
    m = morphometrics(df)
    assert m["total_length"] == pytest.approx(10 + 10 + 2 * np.sqrt(200))
    assert m["n_branch_points"] == 1 and m["n_tips"] == 2 and m["n_stems"] == 1
    s = sholl(df, step=15)
    assert s.loc[s.radius == 15, "intersections"].item() == 1
    assert s.loc[s.radius == 30, "intersections"].item() == 2
