"""Command-line interface: ``neurostias <command>``."""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

app = typer.Typer(add_completion=False, help="NeuroSTIAS - neuroscience analysis system (3D spatial transcriptomics + multi-modal).")
skills_app = typer.Typer(help="List and inspect analysis skills.")
app.add_typer(skills_app, name="skills")
console = Console()

ROOT = Path(__file__).resolve().parents[1]


def _parse_kv(pairs: List[str]) -> dict:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise typer.BadParameter(f"Expected key=value, got '{p}'")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@app.command()
def doctor():
    """Check the environment, optional extras, skills, data and the optional LLM plugin."""
    from neurostias.core import registry
    from neurostias.core.provenance import hardware, package_versions

    hw = hardware()
    console.print(f"[bold]NeuroSTIAS doctor[/bold]  python {hw['python']} on {hw.get('cpu', hw['machine'])}")
    t = Table("group", "package", "status")
    groups = {
        "core": ["numpy", "scipy", "pandas", "sklearn", "anndata", "scanpy", "squidpy", "statsmodels", "networkx"],
        "ephys": ["mne", "elephant", "neo", "pynwb"],
        "modeling": ["brian2"],
        "morphology": ["neurom"],
        "imaging": ["nilearn"],
        "benchmarks": ["SpaGCN", "paste", "celltypist", "liana", "banksy"],
        "llm plugin": ["openai", "anthropic"],
    }
    ok_core = True
    for g, mods in groups.items():
        for m in mods:
            found = importlib.util.find_spec(m) is not None
            if g == "core" and not found:
                ok_core = False
            t.add_row(g, m, "[green]ok[/green]" if found else ("[red]MISSING[/red]" if g == "core" else "[yellow]not installed[/yellow]"))
    console.print(t)

    skills = registry.discover()
    st = Table("skill", "category", "status")
    for name, s in sorted(skills.items()):
        miss = s.missing_requirements()
        st.add_row(name, s.category, "[green]ready[/green]" if not miss else f"[yellow]needs {', '.join(miss)}[/yellow]")
    console.print(st)

    manifest = ROOT / "data" / "manifest.yaml"
    if manifest.exists():
        from neurostias.data import status as data_status
        dt = Table("dataset", "status", "path")
        for row in data_status():
            dt.add_row(row["id"], "[green]present[/green]" if row["present"] else "[yellow]not fetched[/yellow]", row["path"])
        console.print(dt)

    try:
        from plugins.llm_assistant.client import plugin_status
        console.print(f"LLM assistant plugin: {plugin_status()}")
    except Exception:
        console.print("LLM assistant plugin: not installed (optional; `pip install -e '.\\[llm]'`)")

    if not ok_core:
        console.print("[red]Core packages missing - run ./install.sh[/red]")
        raise typer.Exit(1)
    console.print("[green]Core environment OK[/green]")


@skills_app.command("list")
def skills_list(category: Optional[str] = typer.Option(None, help="Filter by category")):
    """List all discovered skills."""
    from neurostias.core import registry
    t = Table("skill", "category", "description")
    for name, s in sorted(registry.discover().items()):
        if category and s.category != category:
            continue
        t.add_row(name, s.category, s.description[:100] + ("..." if len(s.description) > 100 else ""))
    console.print(t)


@skills_app.command("show")
def skills_show(name: str):
    """Show a skill's inputs and outputs."""
    from neurostias.core import registry
    s = registry.get(name)
    console.print(f"[bold]{s.name}[/bold] ({s.category})\n{s.description}\n")
    t = Table("input", "type", "required", "default", "description")
    for i in s.inputs:
        t.add_row(i.name, i.type, str(i.required), str(i.default), i.description + (f" {i.choices}" if i.choices else ""))
    console.print(t)
    if s.references:
        console.print("References: " + "; ".join(s.references))


@app.command()
def run(
    skill: str = typer.Argument(..., help="Skill name, e.g. core.inspect"),
    params: List[str] = typer.Argument(None, help="Parameters as key=value"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory"),
    seed: int = typer.Option(0, help="Random seed"),
):
    """Run one skill, e.g. `neurostias run core.inspect input=data.h5ad`."""
    from neurostias.core.runner import run_skill
    result = run_skill(skill, _parse_kv(params or []), out_dir=out, seed=seed)
    console.print_json(json.dumps({k: v for k, v in result.items() if k != "summary"} | ({"summary": result["summary"]} if "summary" in result else {}), default=str))


@app.command()
def pipeline(
    input: Path = typer.Argument(..., help="Input .h5ad"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
    section_key: Optional[str] = typer.Option(None, help="obs column with section IDs"),
    block_key: Optional[str] = typer.Option(None, help="obs column with animal/donor IDs"),
    z_scale: float = typer.Option(1.0, help="0 = treat sections as independent 2D"),
    n_domains: int = typer.Option(7),
    steps: Optional[str] = typer.Option(None, help="Comma-separated subset of steps"),
    seed: int = typer.Option(0),
):
    """Run the full analysis suite and build a PDF report."""
    from neurostias.pipeline import run_pipeline
    res = run_pipeline(str(input), out_dir=out, steps=steps.split(",") if steps else None, section_key=section_key,
                       block_key=block_key, z_scale=z_scale, n_domains=n_domains, seed=seed)
    for s in res["steps"]:
        console.print(f"{s['step']:<15} {s['status']:<10} {s.get('error', '')}")
    console.print(f"Report: {res['pdf']}")


@app.command()
def data(
    action: str = typer.Argument("list", help="list | fetch | lock | verify"),
    dataset: Optional[str] = typer.Argument(None, help="Dataset id from data/manifest.yaml"),
):
    """List, download, checksum-lock or verify datasets declared in data/manifest.yaml."""
    from neurostias import data as d
    if action == "list":
        t = Table("dataset", "present", "size", "description")
        for row in d.status():
            t.add_row(row["id"], "yes" if row["present"] else "no", row.get("size", ""), row.get("description", "")[:80])
        console.print(t)
    elif action == "fetch":
        if not dataset:
            raise typer.BadParameter("fetch needs a dataset id (or 'all')")
        d.fetch(dataset)
    elif action == "lock":
        d.lock_files(dataset or "all")
    elif action == "verify":
        for row in d.verify(dataset):
            console.print(row)
    else:
        raise typer.BadParameter(action)


@app.command()
def experiment(eid: str = typer.Argument(..., help="Experiment id, e.g. E01, or 'all'")):
    """Run a benchmark experiment from experiments/."""
    import runpy
    exp_root = ROOT / "experiments"
    targets = sorted(exp_root.glob("E*_*/run.py")) if eid == "all" else sorted(exp_root.glob(f"{eid}_*/run.py"))
    if not targets:
        console.print(f"[red]No experiment matching {eid}[/red]")
        raise typer.Exit(1)
    for t in targets:
        console.print(f"[bold]Running {t.parent.name}[/bold]")
        runpy.run_path(str(t), run_name="__main__")


@app.command()
def report():
    """Build paper tables and figures from experiment results."""
    from neurostias import paper
    paper.build()


@app.command()
def ask(
    question: str = typer.Argument(...),
    data_path: Optional[Path] = typer.Option(None, "--data"),
    provider: Optional[str] = typer.Option(None, help="local | anthropic | openai"),
    model: Optional[str] = typer.Option(None),
    endpoint: Optional[str] = typer.Option(None),
):
    """Ask a question in natural language (optional LLM assistant plugin)."""
    try:
        from plugins.llm_assistant.assistant import ask as _ask
    except ImportError:
        console.print("The LLM assistant is an optional plugin. Enable it with:\n  pip install -e '.\\[llm]'\n"
                      "then configure configs/llm.yaml (local LM Studio/Ollama by default).")
        raise typer.Exit(1)
    console.print(_ask(question, data_path=data_path, provider=provider, model=model, endpoint=endpoint))


@app.command()
def menu():
    """Interactive menu (English) for users who prefer not to type commands."""
    from neurostias.menu import main_menu
    main_menu()


if __name__ == "__main__":
    app()
