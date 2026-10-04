"""OpenAI-compatible chat-completions backend (``openai`` SDK).

Used for local servers (LM Studio, Ollama, llama.cpp, local vLLM) and, when
explicitly selected, the OpenAI cloud API.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

from .base import AgentResult, Provider, ToolSpec

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def is_local(url: str | None) -> bool:
    return bool(url) and (urlparse(url).hostname or "") in LOCAL_HOSTS


class OpenAICompatProvider(Provider):
    def __init__(self, model: str, endpoint: str | None = None, api_key: str | None = None, cloud: bool = False):
        from openai import OpenAI
        if not cloud and not is_local(endpoint):
            raise ValueError(f"provider 'local' requires a localhost endpoint, got {endpoint!r}; "
                             "use --provider openai for a cloud endpoint")
        self.is_cloud = cloud
        self.model = model
        self.client = OpenAI(base_url=endpoint, api_key=api_key or ("local" if not cloud else None))

    def run(self, system, user, tools: list[ToolSpec], execute, max_steps=8) -> AgentResult:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        schema = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                    "parameters": t.parameters}} for t in tools]
        calls = []
        for _ in range(max_steps):
            resp = self.client.chat.completions.create(model=self.model, messages=msgs, tools=schema, tool_choice="auto")
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return AgentResult(text=msg.content or "", tool_calls=calls, stop_reason=resp.choices[0].finish_reason,
                                   model=self.model)
            msgs.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    result = f"ERROR: tool arguments were not valid JSON ({exc})"
                    args = {}
                else:
                    result = execute(tc.function.name, args)
                calls.append({"name": tc.function.name, "arguments": args, "result": result})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return AgentResult(text="Stopped: maximum number of tool steps reached.", tool_calls=calls,
                           stop_reason="max_steps", model=self.model)
