import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from neurostias.core import registry, stats
from neurostias.core.provenance import Provenance


def test_registry_discovers_builtin_skills():
    skills = registry.discover()
    assert "core.inspect" in skills
    s = skills["core.inspect"]
    schema = s.to_tool_schema()
    assert schema["parameters"]["required"] == ["input"]


def test_coerce_params_rejects_unknown_and_missing():
    s = registry.get("core.inspect")
    with pytest.raises(ValueError):
        s.coerce_params({})
    with pytest.raises(ValueError):
        s.coerce_params({"input": "x", "bogus": 1})


def test_dropin_skill(tmp_path, monkeypatch):
    d = tmp_path / "my_skill"
    d.mkdir()
    (d / "skill.yaml").write_text(
        "name: user.double\ndescription: doubles a number\nentry: impl.py:run\n"
        "inputs:\n  - {name: x, type: int, required: true}\n")
    (d / "impl.py").write_text("def run(ctx, x):\n    return {'value': 2 * x}\n")
    monkeypatch.setenv("NEUROSTIAS_SKILL_PATH", str(tmp_path))
    from neurostias.core.runner import run_skill
    res = run_skill("user.double", {"x": "21"}, out_dir=tmp_path / "out")
    assert res["value"] == 42
    prov = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert prov["params"] == {"x": 21} and prov["data_origin"] == "real"


def test_provenance_rejects_bad_origin():
    with pytest.raises(ValueError):
        Provenance("skill", "x", data_origin="made_up")


def test_pseudobulk_paired_null_is_calibrated():
    """Under the null (labels independent of expression), paired pseudobulk p-values are ~uniform."""
    rng = np.random.default_rng(0)
    n_units, cells, genes = 6, 200, 300
    X, g, u = [], [], []
    for unit in range(n_units):
        unit_effect = rng.gamma(2, 1, genes)  # strong unit (animal) effect, no group effect
        for grp in ("A", "B"):
            X.append(rng.poisson(unit_effect, size=(cells, genes)))
            g += [grp] * cells
            u += [f"m{unit}"] * cells
    X = sparse.csr_matrix(np.vstack(X))
    res = stats.pseudobulk_test(X, [f"g{i}" for i in range(genes)], pd.Series(g), pd.Series(u), "A", "B")
    assert res["test"].iloc[0] == "paired_t_pseudobulk"
    assert (res["pval"] < 0.05).mean() < 0.12


def test_pseudobulk_unreplicated_returns_no_pvalue():
    X = sparse.csr_matrix(np.ones((40, 5)))
    res = stats.pseudobulk_test(X, list("abcde"), pd.Series(["A"] * 20 + ["B"] * 20), pd.Series(["m1"] * 40), "A", "B")
    assert res["test"].iloc[0] == "insufficient_replicates"
    assert res["pval"].isna().all()
