"""LLM plugin tests with mocked providers/clients (no network, no API key)."""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")
pytest.importorskip("anthropic")

from plugins.llm_assistant import guardrails, privacy  # noqa: E402
from plugins.llm_assistant.providers.base import ToolSpec  # noqa: E402
from plugins.llm_assistant.providers.openai_compat import OpenAICompatProvider, is_local  # noqa: E402


def test_guardrail_flags_unsupported_numbers():
    tool = json.dumps({"n_significant": 711, "ari": 0.83612, "fraction": 0.7457})
    ok = guardrails.check("We found 711 genes (ARI 0.84; 74.6% unresolved).", [tool])
    assert ok["grounded"], ok
    bad = guardrails.check("We found 812 genes with ARI 0.91.", [tool])
    assert set(bad["unverified"]) == {"812", "0.91"}


def test_privacy_redacts_home_and_truncates():
    res = {"out_dir": privacy.HOME + "/x/run", "artifacts": ["a.csv"], "top": list(range(100))}
    s = json.loads(privacy.sanitize_result(res, cloud=True).replace(" ... [truncated]", ""))
    assert "artifacts" not in s and s["out_dir"].startswith("~") and len(s["top"]) == privacy.MAX_LIST + 1


def test_local_provider_refuses_remote_endpoint():
    assert is_local("http://localhost:1234/v1") and not is_local("https://api.example.com/v1")
    with pytest.raises(ValueError):
        OpenAICompatProvider("m", "https://api.example.com/v1")


def test_cloud_requires_acknowledgement(monkeypatch):
    monkeypatch.delenv("NEUROSTIAS_ALLOW_CLOUD", raising=False)
    assert not privacy.cloud_allowed({"cloud": {"acknowledged": False}})
    monkeypatch.setenv("NEUROSTIAS_ALLOW_CLOUD", "1")
    assert privacy.cloud_allowed({})


def test_anthropic_provider_loop_and_request_shape(monkeypatch):
    from plugins.llm_assistant.providers import anthropic_provider as ap
    calls = []

    def fake_create(**kw):
        calls.append(kw)
        if len(calls) == 1:
            return SimpleNamespace(stop_reason="tool_use", model="claude-opus-5-5", content=[
                SimpleNamespace(type="tool_use", id="t1", name="core__inspect", input={"input": "x.h5ad"})])
        return SimpleNamespace(stop_reason="end_turn", model="claude-opus-5-5",
                               content=[SimpleNamespace(type="text", text="Done: 4384 cells.")])

    p = ap.AnthropicProvider.__new__(ap.AnthropicProvider)
    p.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=fake_create)))
    p.model, p.effort, p.use_fallbacks = "claude-opus-5-5", "medium", True
    r = p.run("sys", "q", [ToolSpec("core__inspect", "d", {"type": "object", "properties": {}})],
              lambda n, a: json.dumps({"n_obs": 4384}))
    assert r.text == "Done: 4384 cells." and len(r.tool_calls) == 1
    first = calls[0]
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert first["tool_choice"] == {"type": "auto"} and first["output_config"] == {"effort": "medium"}
    assert calls[1]["messages"][-1]["content"][0]["type"] == "tool_result"


def test_assistant_end_to_end_with_fake_provider(monkeypatch, tmp_path):
    from plugins.llm_assistant import assistant

    class Fake:
        is_cloud = False

        def run(self, system, user, tools, execute, max_steps=8):
            assert any(t.name == "modeling__hh" for t in tools)
            out = execute("modeling__hh", {"currents": "5,10", "T": 200})
            rate = json.loads(out)["rates_hz"]["10.0"]
            from plugins.llm_assistant.providers.base import AgentResult
            return AgentResult(text=f"At 10 uA/cm2 the neuron fires at {rate} Hz.", tool_calls=[{}], model="fake")

    monkeypatch.setattr(assistant, "make_provider", lambda *a, **k: Fake())
    text = assistant.ask("What is the HH rate at 10?", out_root=tmp_path)
    assert "WARNING" not in text and "fake" in text
