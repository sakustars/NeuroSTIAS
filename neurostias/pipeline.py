"""Full analysis pipeline (the NeuroSTIAS equivalent of V4.4.1 "run all modules").

Runs each step as a separate skill run (own folder, results.json, provenance.json)
under one parent folder, passes the updated AnnData from step to step, records
failures without stopping the pipeline, and builds a PDF report from the
figures and results. Steps whose data requirements are not met are reported as
"not applicable", never filled with placeholders.
"""

from __future__ import annotations

import datetime as _dt
import json
import traceback
from pathlib import Path

from .core.runner import run_skill

DEFAULT_STEPS = ["preprocess", "annotate", "overview", "domains", "svg", "niches", "communication",
                 "development", "grn", "metabolic", "deconvolution"]


def run_pipeline(input: str, out_dir: str | Path | None = None, steps: list[str] | None = None,
                 section_key: str | None = None, block_key: str | None = None, z_scale: float = 1.0,
                 n_domains: int = 7, skip_preprocess_if_processed: bool = True, seed: int = 0) -> dict:
    steps = steps or DEFAULT_STEPS
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(out_dir) if out_dir else Path.cwd() / "runs" / f"pipeline_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    current = str(input)
    log = []

    def step(name, skill, params, updates_h5ad=False):
        nonlocal current
        try:
            res = run_skill(skill, {"input": current, **params}, out_dir=root / f"{len(log) + 1:02d}_{name}", seed=seed)
            if updates_h5ad and res.get("h5ad"):
                current = res["h5ad"]
            log.append({"step": name, "skill": skill, "status": "completed", "out_dir": res["out_dir"],
                        "elapsed_seconds": res["elapsed_seconds"]})
        except Exception as exc:  # noqa: BLE001 - record and continue
            log.append({"step": name, "skill": skill, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=3)})

    from .core.io import read_h5ad
    a = read_h5ad(current, backed=True)
    processed = "log_normalized" in a.layers and "X_pca" in a.obsm
    a.file.close()
    sk = {"section_key": section_key} if section_key else {}
    bk = {"block_key": block_key} if block_key else {}
    for s in steps:
        if s == "preprocess":
            if processed and skip_preprocess_if_processed:
                log.append({"step": s, "status": "skipped", "reason": "input already preprocessed"})
            else:
                step(s, "core.preprocess", {}, True)
        elif s == "annotate":
            step(s, "atlas.annotate", {**sk}, True)
        elif s == "overview":
            step(s, "legacy.misc", {"analysis": "overview"})
        elif s == "domains":
            step(s, "spatial3d.domains", {"n_domains": n_domains, "z_scale": z_scale, **sk, **bk}, True)
        elif s == "svg":
            step(s, "spatial3d.svg", {"z_scale": z_scale, **sk, **bk})
        elif s == "niches":
            step(s, "spatial3d.niches", {"z_scale": z_scale, **sk, **bk}, True)
        elif s == "communication":
            step(s, "spatial3d.communication", {"z_scale": z_scale, **sk, **bk})
        elif s == "development":
            step(s, "legacy.development", {**sk}, True)
        elif s == "grn":
            step(s, "legacy.grn", {"context_key": "neuro_class"})
        elif s in ("metabolic", "deconvolution", "gcn", "colocalization", "intervention"):
            step(s, "legacy.misc", {"analysis": s, **({"group_key": "neuro_class"} if s == "metabolic" else {})})
        else:
            log.append({"step": s, "status": "unknown_step"})
    (root / "pipeline_log.json").write_text(json.dumps(log, indent=2))
    pdf = build_pdf(root, title="NeuroSTIAS analysis report", input_path=str(input))
    return {"out_dir": str(root), "final_h5ad": current, "steps": log, "pdf": str(pdf)}


def build_pdf(root: Path, title: str, input_path: str = "") -> Path:
    """Assemble a PDF: title page, step table, then each step's figures and key results."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    root = Path(root)
    log = json.loads((root / "pipeline_log.json").read_text()) if (root / "pipeline_log.json").exists() else []
    pdf_path = root / "report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.92, title, fontsize=18, weight="bold")
        fig.text(0.08, 0.88, f"Input: {input_path}", fontsize=8)
        fig.text(0.08, 0.86, f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}", fontsize=8)
        y = 0.80
        for row in log:
            line = f"{row['step']:<16} {row['status']:<10} {row.get('error', row.get('reason', ''))[:90]}"
            fig.text(0.08, y, line, fontsize=8, family="monospace")
            y -= 0.022
        fig.text(0.08, 0.06, "Every number in this report comes from the results.json / CSV files in the run folder; "
                             "provenance.json in each step folder records inputs, parameters and versions.", fontsize=7)
        pdf.savefig(fig)
        plt.close(fig)
        for row in log:
            if row.get("status") != "completed":
                continue
            d = Path(row["out_dir"])
            res = json.loads((d / "results.json").read_text())
            brief = {k: v for k, v in res.items() if k not in ("artifacts", "out_dir", "h5ad", "summary", "top", "coverage")}
            for png in sorted(d.rglob("*.png")):
                img = mpimg.imread(png)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.text(0.06, 0.96, f"{row['step']} - {png.name}", fontsize=11, weight="bold")
                ax = fig.add_axes([0.05, 0.30, 0.9, 0.63])
                ax.imshow(img)
                ax.axis("off")
                fig.text(0.06, 0.27, json.dumps(brief, indent=1, default=str)[:2500], fontsize=6, family="monospace",
                         va="top")
                pdf.savefig(fig)
                plt.close(fig)
    return pdf_path
