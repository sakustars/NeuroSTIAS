"""Run a skill with a uniform context: output folder, seed, provenance and logging."""

from __future__ import annotations

import datetime as _dt
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import registry
from .provenance import Provenance, write_json

FILE_TYPES = {"h5ad", "path", "csv", "nwb", "swc", "nifti", "fif", "edf"}
DEFAULT_RUNS_DIR = Path.cwd() / "runs"


@dataclass
class RunContext:
    out_dir: Path
    seed: int
    provenance: Provenance
    log: logging.Logger
    artifacts: list[str] = field(default_factory=list)

    def path(self, name: str) -> Path:
        p = self.out_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts.append(str(p.relative_to(self.out_dir)))
        return p

    def save_figure(self, fig, name: str, dpi: int = 200) -> Path:
        p = self.path(name)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return p

    def save_table(self, df, name: str, index: bool = True) -> Path:
        p = self.path(name)
        df.to_csv(p, index=index)
        return p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # optional
        torch.manual_seed(seed)
    except Exception:
        pass


def _logger(out_dir: Path) -> logging.Logger:
    log = logging.getLogger(f"neurostias.run.{out_dir.name}")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fh = logging.FileHandler(out_dir / "run.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(sh)
    return log


def run_skill(name: str, params: dict[str, Any] | None = None, out_dir: str | Path | None = None,
              seed: int = 0, data_origin: str = "real") -> dict[str, Any]:
    skill = registry.get(name)
    missing = skill.missing_requirements()
    if missing:
        raise RuntimeError(f"Skill {name} needs optional package(s) {missing}. "
                           f"Install with: pip install -e '.[all]'")
    clean = skill.coerce_params(params or {})
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir) if out_dir else DEFAULT_RUNS_DIR / f"{name}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    prov = Provenance("skill", name, params=clean, seed=seed, data_origin=data_origin)
    for spec in skill.inputs:
        val = clean.get(spec.name)
        if spec.type in FILE_TYPES and val and Path(str(val)).exists():
            prov.add_input(val, label=spec.name)
    ctx = RunContext(out_dir=out, seed=seed, provenance=prov, log=_logger(out))
    seed_everything(seed)
    func = skill.load()
    t0 = time.perf_counter()
    result = func(ctx=ctx, **clean) if registry.accepts_context(func) else func(**clean)
    elapsed = time.perf_counter() - t0
    result = dict(result or {})
    result.setdefault("skill", name)
    result["elapsed_seconds"] = round(elapsed, 3)
    result["out_dir"] = str(out)
    result["artifacts"] = ctx.artifacts
    write_json(result, out / "results.json")
    prov.write(out, extra={"elapsed_seconds": elapsed, "artifacts": ctx.artifacts})
    return result
