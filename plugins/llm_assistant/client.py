"""Provider factory and plugin status."""

from __future__ import annotations

import importlib.util
import os
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    p = ROOT / "configs" / "llm.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {"provider": "local", "local": {}}


def make_provider(cfg: dict, provider: str | None = None, model: str | None = None, endpoint: str | None = None):
    name = provider or cfg.get("provider", "local")
    if name == "local":
        from .providers.openai_compat import OpenAICompatProvider
        c = cfg.get("local", {})
        return OpenAICompatProvider(model or c.get("model"), endpoint or c.get("endpoint", "http://localhost:1234/v1"))
    if name == "openai":
        from .providers.openai_compat import OpenAICompatProvider
        c = cfg.get("openai", {})
        return OpenAICompatProvider(model or c.get("model"), endpoint or c.get("endpoint"),
                                    api_key=os.environ.get("OPENAI_API_KEY"), cloud=True)
    if name == "anthropic":
        from .providers.anthropic_provider import AnthropicProvider
        c = cfg.get("anthropic", {})
        return AnthropicProvider(model or c.get("model", "claude-opus-5-5"), effort=c.get("effort", "medium"))
    raise ValueError(f"unknown provider {name!r} (local | anthropic | openai)")


def plugin_status() -> str:
    cfg = load_config()
    if importlib.util.find_spec("openai") is None:
        return "installed but 'openai' package missing (pip install -e '.[llm]')"
    prov = cfg.get("provider", "local")
    if prov == "local":
        ep = cfg.get("local", {}).get("endpoint", "http://localhost:1234/v1")
        try:
            with urllib.request.urlopen(ep.rstrip("/") + "/models", timeout=2) as r:
                ok = r.status == 200
        except Exception:
            ok = False
        return f"enabled (local: {ep}, model {cfg['local'].get('model')}; server {'reachable' if ok else 'NOT reachable'})"
    return f"enabled (cloud provider: {prov}; model {cfg.get(prov, {}).get('model')})"
