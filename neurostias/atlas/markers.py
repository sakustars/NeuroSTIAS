"""Load curated brain marker panels for a given species."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

MARKER_FILE = Path(__file__).resolve().parent / "markers" / "brain_markers.yaml"
SECTIONS = ("classes", "neuron_subclasses", "neurotransmitter", "cortical_layers", "developing_regions",
            "mouse_brain_regions", "states", "neurodevelopment")


def _to_mouse(symbol: str) -> str:
    return symbol[:1].upper() + symbol[1:].lower()


@lru_cache(maxsize=None)
def _raw() -> dict:
    return yaml.safe_load(MARKER_FILE.read_text())


def panels(section: str, species: str = "human") -> dict[str, list[str]]:
    """Return {panel_name: [genes]} for one section and species ('human' or 'mouse')."""
    if section not in SECTIONS and section != "pan_neuronal":
        raise KeyError(f"Unknown marker section '{section}'. Options: {SECTIONS}")
    data = _raw()[section]
    if section == "pan_neuronal":
        data = {"Neuron": data}
    out = {}
    for name, spec in data.items():
        if species == "mouse":
            genes = spec.get("mouse") or [_to_mouse(g) for g in spec.get("human", [])]
        else:
            genes = spec.get("human") or []
        if genes:
            out[name] = list(genes)
    return out


def parents() -> dict[str, str]:
    return {k: v["parent"] for k, v in _raw()["neuron_subclasses"].items()}


def all_genes(species: str = "human") -> set[str]:
    genes: set[str] = set()
    for s in SECTIONS:
        for g in panels(s, species).values():
            genes.update(g)
    return genes


def version() -> int:
    return int(_raw().get("version", 0))
