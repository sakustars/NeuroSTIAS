"""Skill registry: discovers analysis skills from ``skill.yaml`` manifests.

A skill is a folder containing ``skill.yaml``::

    name: atlas.annotate
    category: atlas
    description: Annotate brain cell types with curated neuroscience marker panels.
    entry: neurostias.atlas.annotate:run_skill
    inputs:
      - {name: input, type: h5ad, required: true, description: AnnData file}
      - {name: species, type: str, default: auto, choices: [auto, human, mouse]}
    outputs: [annotations.csv, summary.png]

Built-in skills live in ``neurostias/skills``. Users can drop extra skill folders
into ``./skills`` (next to the project) or any directory listed in the
``NEUROSTIAS_SKILL_PATH`` environment variable; they are picked up
automatically. The entry may also be ``file.py:function`` relative to the
skill folder, so a drop-in skill does not need to be an installed package.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

PACKAGE_SKILLS = Path(__file__).resolve().parents[1] / "skills"
PROJECT_SKILLS = Path(__file__).resolve().parents[2] / "skills"


@dataclass
class SkillInput:
    name: str
    type: str = "str"
    required: bool = False
    default: Any = None
    description: str = ""
    choices: list | None = None


@dataclass
class Skill:
    name: str
    category: str
    description: str
    entry: str
    folder: Path
    inputs: list[SkillInput] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def load(self) -> Callable:
        target, func = self.entry.split(":")
        if target.endswith(".py"):
            spec = importlib.util.spec_from_file_location(f"neurostias_skill_{self.name.replace('.', '_')}",
                                                          self.folder / target)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        else:
            mod = importlib.import_module(target)
        return getattr(mod, func)

    def missing_requirements(self) -> list[str]:
        return [r for r in self.requires if importlib.util.find_spec(r) is None]

    def coerce_params(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Apply defaults, check required inputs and choices, convert simple types."""
        out: dict[str, Any] = {}
        known = {i.name for i in self.inputs}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown parameter(s) for {self.name}: {sorted(unknown)}")
        for spec in self.inputs:
            if spec.name in raw and raw[spec.name] is not None:
                val = raw[spec.name]
            elif spec.required:
                raise ValueError(f"Missing required parameter '{spec.name}' for skill {self.name}")
            else:
                val = spec.default
            if val is not None:
                val = _convert(val, spec.type)
                if spec.choices and val not in spec.choices:
                    raise ValueError(f"{spec.name}={val!r} not in {spec.choices}")
            out[spec.name] = val
        return out

    def to_tool_schema(self) -> dict:
        """JSON schema used by the optional LLM plugin for tool calling."""
        props, required = {}, []
        type_map = {"int": "integer", "float": "number", "bool": "boolean", "list": "array"}
        for i in self.inputs:
            p: dict[str, Any] = {"type": type_map.get(i.type, "string"), "description": i.description}
            if i.choices:
                p["enum"] = i.choices
            if i.type == "list":
                p["items"] = {"type": "string"}
            props[i.name] = p
            if i.required:
                required.append(i.name)
        return {"name": self.name.replace(".", "__"), "description": self.description,
                "parameters": {"type": "object", "properties": props, "required": required}}


def _convert(val: Any, typ: str) -> Any:
    if typ == "int":
        return int(val)
    if typ == "float":
        return float(val)
    if typ == "bool":
        if isinstance(val, str):
            return val.strip().lower() in {"1", "true", "yes", "y"}
        return bool(val)
    if typ == "list":
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        return list(val)
    return val


def skill_dirs() -> list[Path]:
    dirs = [PACKAGE_SKILLS, PROJECT_SKILLS]
    for extra in os.environ.get("NEUROSTIAS_SKILL_PATH", "").split(os.pathsep):
        if extra:
            dirs.append(Path(extra).expanduser())
    return [d for d in dirs if d.is_dir()]


def _parse(manifest: Path) -> Skill:
    data = yaml.safe_load(manifest.read_text())
    for key in ("name", "description", "entry"):
        if key not in data:
            raise ValueError(f"{manifest}: missing '{key}'")
    return Skill(
        name=data["name"],
        category=data.get("category", data["name"].split(".")[0]),
        description=" ".join(str(data["description"]).split()),
        entry=data["entry"],
        folder=manifest.parent,
        inputs=[SkillInput(**i) for i in data.get("inputs", [])],
        outputs=list(data.get("outputs", [])),
        requires=list(data.get("requires", [])),
        references=list(data.get("references", [])),
    )


def discover() -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    for d in skill_dirs():
        for manifest in sorted(d.glob("*/skill.yaml")):
            s = _parse(manifest)
            skills[s.name] = s  # later directories override built-ins on purpose
    return skills


def get(name: str) -> Skill:
    skills = discover()
    if name not in skills:
        raise KeyError(f"Unknown skill '{name}'. Available: {', '.join(sorted(skills))}")
    return skills[name]


def accepts_context(func: Callable) -> bool:
    return "ctx" in inspect.signature(func).parameters
