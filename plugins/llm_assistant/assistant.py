"""`neurostias ask`: answer a question by calling NeuroSTIAS skills through any configured LLM."""

from __future__ import annotations

import json
from pathlib import Path

from neurostias.core import registry
from neurostias.core.runner import run_skill

from . import guardrails, privacy
from .client import load_config, make_provider
from .providers.base import ToolSpec

SYSTEM = """You are the NeuroSTIAS assistant for neuroscience data analysis.
- Answer questions about the user's data only by calling the provided analysis tools; never invent results.
- Every number you report about data must come from a tool result. If a tool reports that an analysis is not
  applicable or a panel is unusable, say so plainly.
- For general neuroscience knowledge (no data needed) answer briefly and say it is background knowledge.
- Mention the output folder (out_dir) of each analysis you ran so the user can inspect files and provenance.
- Prefer statistically conservative interpretations; spots/cells from one animal are not independent replicates."""


def _tools(include_docs: bool) -> tuple[list[ToolSpec], dict]:
    specs, names = [], {}
    for name, s in sorted(registry.discover().items()):
        if s.missing_requirements():
            continue
        sch = s.to_tool_schema()
        specs.append(ToolSpec(sch["name"], sch["description"], sch["parameters"]))
        names[sch["name"]] = name
    if include_docs:
        specs.append(ToolSpec("search_docs", "Search the user's local reference documents (textbooks, papers) for passages.",
                              {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}))
    return specs, names


def ask(question: str, data_path: str | Path | None = None, provider: str | None = None, model: str | None = None,
        endpoint: str | None = None, docs: str | Path | None = None, out_root: str | Path | None = None) -> str:
    cfg = load_config()
    prov = make_provider(cfg, provider, model, endpoint)
    if prov.is_cloud and not privacy.cloud_allowed(cfg):
        return privacy.CLOUD_NOTICE
    index = None
    if docs:
        from .rag import DocIndex
        index = DocIndex(docs)
    specs, names = _tools(index is not None)
    results_for_check: list[str] = []
    runs_dir = Path(out_root) if out_root else Path.cwd() / "runs"

    def execute(tool: str, args: dict) -> str:
        try:
            if tool == "search_docs":
                out = json.dumps(index.search(args.get("query", "")))
            else:
                res = run_skill(names[tool], args, out_dir=None if out_root is None else runs_dir / f"{tool}_{len(results_for_check)}")
                out = privacy.sanitize_result(res, prov.is_cloud)
        except Exception as exc:  # noqa: BLE001 - errors go back to the model
            out = f"ERROR: {type(exc).__name__}: {exc}"
        results_for_check.append(out)
        return out

    user = question if not data_path else f"{question}\n\nDataset: {data_path}"
    result = prov.run(SYSTEM, user, specs, execute, max_steps=int(cfg.get("max_steps", 8)))
    check = guardrails.check(result.text, results_for_check, user)
    lines = [result.text.strip(), "", f"[model: {result.model}; tool calls: {len(result.tool_calls)}; "
             f"numbers checked: {check['n_numbers_checked']}]"]
    if check["unverified"]:
        lines.append("WARNING - numbers not traceable to any tool result: " + ", ".join(check["unverified"]))
    return "\n".join(lines)
