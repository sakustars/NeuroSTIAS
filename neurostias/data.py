"""Dataset manifest: declared sources, local presence, download and checksum verification.

``data/manifest.yaml`` lists every dataset used by skills and experiments with
its source URL, licence and citation. Downloaded files land in ``data/<id>/``.
The SHA-256 of each file is recorded in ``data/checksums.lock.json`` on first
download, so later runs (and other machines) can verify they use identical bytes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

from .core.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NEUROSTIAS_DATA", ROOT / "data"))
MANIFEST = ROOT / "data" / "manifest.yaml"
LOCK = ROOT / "data" / "checksums.lock.json"


def manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    return yaml.safe_load(MANIFEST.read_text()).get("datasets", {})


def dataset_dir(ds_id: str) -> Path:
    entry = manifest().get(ds_id, {})
    if entry.get("local_path"):
        return Path(os.path.expanduser(entry["local_path"]))
    return DATA_DIR / ds_id


def path(ds_id: str, file: str | None = None) -> Path:
    """Resolve a dataset (or a file inside it). Raises with a fetch hint if missing."""
    entry = manifest().get(ds_id)
    if entry is None:
        raise KeyError(f"Dataset '{ds_id}' not in data/manifest.yaml")
    base = dataset_dir(ds_id)
    target = base / (file or entry.get("main_file", ""))
    if not target.exists():
        raise FileNotFoundError(f"{target} not found. Run: neurostias data fetch {ds_id}")
    return target


def _present(ds_id: str, entry: dict) -> bool:
    base = dataset_dir(ds_id)
    main = entry.get("main_file")
    return (base / main).exists() if main else base.exists() and any(base.iterdir())


def status() -> list[dict]:
    rows = []
    for ds_id, entry in manifest().items():
        rows.append({
            "id": ds_id,
            "present": _present(ds_id, entry),
            "path": str(dataset_dir(ds_id)),
            "size": entry.get("approx_size", ""),
            "description": entry.get("description", ""),
        })
    return rows


def _load_lock() -> dict:
    return json.loads(LOCK.read_text()) if LOCK.exists() else {}


def _save_lock(lock: dict) -> None:
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}\n    -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "neurostias/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if total:
                sys.stdout.write(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB")
                sys.stdout.flush()
    sys.stdout.write("\n")
    tmp.rename(dest)


def fetch(ds_id: str) -> None:
    entries = manifest()
    targets = list(entries) if ds_id == "all" else [ds_id]
    lock = _load_lock()
    for t in targets:
        entry = entries[t]
        base = dataset_dir(t)
        if entry.get("local_path"):
            print(f"{t}: local dataset at {base} ({'present' if _present(t, entry) else 'MISSING'})")
            continue
        print(f"{t}: {entry.get('description', '')}")
        for f in entry.get("files", []):
            dest = base / f["name"]
            if not dest.exists():
                _download(f["url"], dest)
            digest = sha256_file(dest)
            expected = f.get("sha256") or lock.get(t, {}).get(f["name"])
            if expected and expected != digest:
                raise RuntimeError(f"Checksum mismatch for {dest}: expected {expected}, got {digest}")
            lock.setdefault(t, {})[f["name"]] = digest
            if f.get("extract"):
                _extract(dest, base)
        if entry.get("script"):
            subprocess.run([sys.executable, str(ROOT / entry["script"]), "--out", str(base)], check=True)
        _save_lock(lock)


def _extract(archive: Path, dest: Path) -> None:
    marker = dest / f".extracted_{archive.name}"
    if marker.exists():
        return
    print(f"  extracting {archive.name}")
    shutil.unpack_archive(str(archive), str(dest))
    marker.touch()


# files that are transient download/cache artefacts, not inputs to any analysis
_LOCK_SKIP_SUFFIXES = (".part", ".curlpart", ".pkl", ".lock", ".log")
_LOCK_SKIP_PARTS = ("cache", "__pycache__", "joblib")


def lock_files(ds_id: str = "all") -> dict:
    """Record the SHA-256 of every file present in a downloaded dataset folder.

    ``fetch`` only locks files listed explicitly in the manifest; datasets produced by
    a fetch script (subsets, extracted time series) are locked here, after the fact,
    so other machines can verify they reproduce identical inputs. Local (user-provided)
    datasets are skipped.
    """
    entries = manifest()
    targets = list(entries) if ds_id == "all" else [ds_id]
    lock = _load_lock()
    for t in targets:
        entry = entries[t]
        base = dataset_dir(t)
        if entry.get("local_path") or not base.exists():
            continue
        for f in sorted(base.rglob("*")):
            rel = f.relative_to(base)
            if (not f.is_file() or f.name.startswith(".") or f.suffix in _LOCK_SKIP_SUFFIXES
                    or "CORRUPT" in f.name or any(s in part for part in rel.parts[:-1] for s in _LOCK_SKIP_PARTS)):
                continue
            lock.setdefault(t, {})[str(rel)] = sha256_file(f)
        print(f"{t}: {len(lock.get(t, {}))} files locked")
    _save_lock(lock)
    return lock


def verify(ds_id: str | None = None) -> list[dict]:
    lock = _load_lock()
    out = []
    for t, files in lock.items():
        if ds_id and t != ds_id:
            continue
        for name, expected in files.items():
            p = dataset_dir(t) / name
            ok = p.exists() and sha256_file(p) == expected
            out.append({"dataset": t, "file": name, "ok": ok})
    return out
