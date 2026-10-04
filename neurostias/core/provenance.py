"""Provenance recording.

Every skill run and every experiment writes a ``provenance.json`` next to its
results. The record is what lets a reader of the paper trace each number back
to an exact input file, code version, parameter set and random seed.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

_HASH_CACHE = Path(os.environ.get("NEUROSTIAS_CACHE", Path.home() / ".cache" / "neurostias")) / "sha256_cache.json"

TRACKED_PACKAGES = (
    "numpy", "scipy", "pandas", "scikit-learn", "anndata", "scanpy", "squidpy",
    "statsmodels", "networkx", "mne", "elephant", "brian2", "neurom", "nilearn",
    "SpaGCN", "paste-bio", "celltypist", "liana", "torch",
)


def _load_hash_cache() -> dict:
    try:
        return json.loads(_HASH_CACHE.read_text())
    except Exception:
        return {}


def _save_hash_cache(cache: dict) -> None:
    try:
        _HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _HASH_CACHE.write_text(json.dumps(cache, indent=1))
    except Exception:
        pass


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    """SHA-256 of a file, cached by (path, size, mtime) so multi-GB inputs are hashed once."""
    path = Path(path).resolve()
    st = path.stat()
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    cache = _load_hash_cache()
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    digest = h.hexdigest()
    cache[key] = digest
    _save_hash_cache(cache)
    return digest


def git_state(repo: str | Path | None = None) -> dict:
    repo = Path(repo or Path(__file__).resolve().parents[2])
    def _run(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {
        "commit": _run("rev-parse", "HEAD") or None,
        "dirty": bool(_run("status", "--porcelain")),
    }


def package_versions(names: Iterable[str] = TRACKED_PACKAGES) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            continue
    return out


def hardware() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0]}
    try:
        if sys.platform == "darwin":
            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            info["mem_bytes"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
        info["n_cpu"] = os.cpu_count()
    except Exception:
        pass
    return info


REPO_ROOT = Path(__file__).resolve().parents[2]


def portable_path(p: str | Path) -> str:
    """Path relative to the repository if inside it, otherwise with the home directory as '~'.

    Keeps provenance meaningful on other machines and avoids recording user names."""
    p = Path(p)
    try:
        rp = p.resolve()
        return str(rp.relative_to(REPO_ROOT))
    except (ValueError, OSError):
        pass
    s = str(p.resolve()) if p.exists() else str(p)
    home = str(Path.home())
    return "~" + s[len(home):] if s.startswith(home) else s


def _jsonable(obj: Any):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return repr(obj)


class Provenance:
    """Collects provenance for one run; call :meth:`write` at the end."""

    def __init__(self, kind: str, name: str, params: dict | None = None, seed: int | None = None,
                 data_origin: str = "real"):
        if data_origin not in {"real", "semi_synthetic", "synthetic"}:
            raise ValueError("data_origin must be 'real', 'semi_synthetic' or 'synthetic'")
        self.record: dict[str, Any] = {
            "kind": kind,
            "name": name,
            "data_origin": data_origin,
            "params": _jsonable(params or {}),
            "seed": seed,
            "inputs": {},
            "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "git": git_state(),
            "packages": package_versions(),
            "hardware": hardware(),
            "notes": [],
        }

    def add_input(self, path: str | Path, label: str | None = None, hash_file: bool = True) -> None:
        p = Path(path)
        entry: dict[str, Any] = {"path": portable_path(p)}
        if p.is_file():
            entry["bytes"] = p.stat().st_size
            if hash_file:
                entry["sha256"] = sha256_file(p)
        self.record["inputs"][label or p.name] = entry

    def note(self, text: str) -> None:
        self.record["notes"].append(text)

    def write(self, out_dir: str | Path, extra: dict | None = None) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.record["finished_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        if extra:
            self.record.update(_jsonable(extra))
        path = out_dir / "provenance.json"
        path.write_text(json.dumps(self.record, indent=2))
        return path


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2))
    return path
