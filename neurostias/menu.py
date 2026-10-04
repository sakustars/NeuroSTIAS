"""Interactive English menu (replaces the V4.4.1 Chinese console menu).

All analysis logic lives in skills; the menu only collects parameters and calls
``run_skill`` / ``run_pipeline``, so menu runs produce the same provenance as CLI runs.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from .core import registry
from .core.runner import run_skill

console = Console()


def _ask_params(skill) -> dict:
    params = {}
    for i in skill.inputs:
        label = f"{i.name} ({i.type}{', required' if i.required else ''})"
        if i.description:
            label += f" - {i.description}"
        default = None if i.default is None else str(i.default)
        if i.type == "bool":
            params[i.name] = Confirm.ask(label, default=bool(i.default))
        elif i.type == "int":
            v = Prompt.ask(label, default=default or "")
            params[i.name] = int(v) if v else None
        elif i.type == "float":
            v = Prompt.ask(label, default=default or "")
            params[i.name] = float(v) if v else None
        else:
            v = Prompt.ask(label, default=default or "", choices=i.choices) if i.choices else Prompt.ask(label, default=default or "")
            params[i.name] = v or None
    return {k: v for k, v in params.items() if v is not None}


def main_menu() -> None:
    while True:
        console.rule("[bold]NeuroSTIAS[/bold] - neuroscience spatial analysis")
        console.print("1  Run the full pipeline on a dataset\n2  Run one analysis (skill)\n3  List skills\n"
                      "4  Load a CSV triplet (expression / coordinates / metadata)\n5  Environment check\n0  Exit")
        choice = Prompt.ask("Choose", choices=["0", "1", "2", "3", "4", "5"], default="1")
        if choice == "0":
            return
        try:
            if choice == "1":
                from .pipeline import run_pipeline
                inp = Prompt.ask("Path to .h5ad")
                sec = Prompt.ask("Section column (blank if none)", default="") or None
                blk = Prompt.ask("Sample/animal column (blank if none)", default="") or None
                z = FloatPrompt.ask("z_scale (0 = treat sections as 2D)", default=1.0)
                nd = IntPrompt.ask("Number of spatial domains", default=7)
                res = run_pipeline(inp, section_key=sec, block_key=blk, z_scale=z, n_domains=nd)
                console.print(f"Done. Report: {res['pdf']}")
            elif choice == "2":
                skills = registry.discover()
                names = sorted(skills)
                t = Table("#", "skill", "description")
                for k, n in enumerate(names, 1):
                    t.add_row(str(k), n, skills[n].description[:80])
                console.print(t)
                k = IntPrompt.ask("Skill number", default=1)
                s = skills[names[k - 1]]
                res = run_skill(s.name, _ask_params(s))
                console.print(f"Done. Results in {res['out_dir']}")
            elif choice == "3":
                from .cli import skills_list
                skills_list(category=None)
            elif choice == "4":
                e = Prompt.ask("Expression CSV")
                c = Prompt.ask("Coordinates CSV")
                m = Prompt.ask("Metadata CSV")
                res = run_skill("core.load_triplet", {"expression": e, "coordinates": c, "metadata": m})
                console.print(f"Saved {res['h5ad']}")
            elif choice == "5":
                from .cli import doctor
                doctor()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {exc}")
