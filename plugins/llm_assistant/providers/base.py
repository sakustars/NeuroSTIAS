"""Provider interface: each backend runs its own tool-calling loop in its own SDK's message format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema


@dataclass
class AgentResult:
    text: str
    tool_calls: list[dict] = field(default_factory=list)  # {"name", "arguments", "result"}
    stop_reason: str = ""
    model: str = ""


class Provider:
    is_cloud: bool = False

    def run(self, system: str, user: str, tools: list[ToolSpec], execute: Callable[[str, dict], str],
            max_steps: int = 8) -> AgentResult:
        raise NotImplementedError
