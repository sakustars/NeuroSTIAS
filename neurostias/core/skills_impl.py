"""Entry points for core skills."""

from __future__ import annotations

from .io import read_h5ad, summarize


def inspect_skill(ctx, input: str) -> dict:
    adata = read_h5ad(input, backed=True)
    summary = summarize(adata)
    ctx.log.info("%s: %d obs x %d vars", input, summary["n_obs"], summary["n_vars"])
    return {"summary": summary}
