"""Anthropic Claude backend (official ``anthropic`` SDK, Messages API manual tool loop).

Defaults: ``claude-opus-5-5``, explicit effort (this model defaults to medium),
``tool_choice`` auto (forced tool choice is not supported on current models), and
server-side refusal fallbacks (``fallbacks="default"``). Credentials come from
ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an ``ant auth login`` profile.
"""

from __future__ import annotations

from .base import AgentResult, Provider, ToolSpec


class AnthropicProvider(Provider):
    is_cloud = True

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", use_fallbacks: bool = True):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.use_fallbacks = use_fallbacks

    def _create(self, **kw):
        if self.use_fallbacks:
            return self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
        return self.client.messages.create(**kw)

    def run(self, system, user, tools: list[ToolSpec], execute, max_steps=8) -> AgentResult:
        schema = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        messages = [{"role": "user", "content": user}]
        calls = []
        response = None
        for _ in range(max_steps):
            response = self._create(model=self.model, max_tokens=16000, system=system, tools=schema,
                                    tool_choice={"type": "auto"}, output_config={"effort": self.effort},
                                    messages=messages)
            if response.stop_reason == "refusal":
                return AgentResult(text="The model declined this request.", tool_calls=calls,
                                   stop_reason="refusal", model=response.model)
            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                text = "".join(b.text for b in response.content if b.type == "text")
                return AgentResult(text=text, tool_calls=calls, stop_reason=response.stop_reason, model=response.model)
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                out = execute(tu.name, dict(tu.input))
                calls.append({"name": tu.name, "arguments": dict(tu.input), "result": out})
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out,
                                "is_error": out.startswith("ERROR")})
            messages.append({"role": "user", "content": results})  # all results in one user message
        return AgentResult(text="Stopped: maximum number of tool steps reached.", tool_calls=calls,
                           stop_reason="max_steps", model=self.model)
