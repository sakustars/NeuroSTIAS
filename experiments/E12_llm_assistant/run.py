"""E12 (optional, plugin) - Does the assistant route questions to the right analysis and avoid unsupported numbers?

Tasks: experiments/E12_llm_assistant/tasks.json (154 questions with gold skill and key parameters,
written before any model was evaluated; 20 are knowledge questions whose correct behaviour
is to call no tool).
Arm 1 - routing (all tasks): the first tool call is recorded and answered with a stub
("analysis completed"), so models are compared only on choosing the skill and parameters.
Arm 2 - grounding (30 data tasks, real execution): every number in the final answer is checked
against tool outputs by the guardrail; reported = fraction of answers with >= 1 untraceable
number, i.e. what a user would see without the guardrail.
Models: whichever local models are passed with --models (LM Studio, localhost only).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neurostias.experiment import Experiment, md_table  # noqa: E402
from plugins.llm_assistant import guardrails  # noqa: E402
from plugins.llm_assistant.assistant import SYSTEM, _tools  # noqa: E402
from plugins.llm_assistant.providers.openai_compat import OpenAICompatProvider  # noqa: E402

GROUND_SKILLS = {"core.inspect", "atlas.annotate", "spatial3d.svg", "modeling.hh", "ephys.spikes", "morphology.swc"}


def norm(v):
    return str(v).strip().lower().rstrip("/")


def main(models, endpoint, n_ground):
    exp = Experiment("E12", "Optional LLM assistant: routing and numeric grounding", "Do local models choose the right "
                     "NeuroSTIAS skill and parameters, and how often do answers contain numbers not traceable to tool "
                     "outputs?", data_origin="real", params={"models": models, "endpoint": endpoint, "n_ground": n_ground})
    tasks = json.loads((exp.dir / "tasks.json").read_text())["items"]
    specs, names = _tools(False)
    rows, grows = [], []
    for model in models:
        prov = OpenAICompatProvider(model, endpoint)
        for t in tasks:
            first = {}

            def stub(tool, args):
                if not first:
                    first.update({"tool": names.get(tool, tool), "args": args})
                return json.dumps({"status": "completed", "message": "analysis completed; results saved to the run folder"})
            t0 = time.time()
            try:
                prov.run(SYSTEM, t["question"], specs, stub, max_steps=2)
                err = ""
            except Exception as exc:  # noqa: BLE001
                err = str(exc)[:200]
            chosen = first.get("tool", "none")
            gp = t["gold_params"]
            pm = all(norm(first.get("args", {}).get(k)) == norm(v) for k, v in gp.items()) if gp else True
            rows.append({"model": model, "id": t["id"], "gold": t["gold_skill"], "chosen": chosen,
                         "skill_correct": chosen == t["gold_skill"], "params_correct": bool(pm and chosen == t["gold_skill"]),
                         "seconds": time.time() - t0, "error": err})
        print(model, "routing done", flush=True)
        from neurostias.core.runner import run_skill
        gtasks = [t for t in tasks if t["gold_skill"] in GROUND_SKILLS][:n_ground]
        for t in gtasks:
            outs = []

            def real(tool, args):
                try:
                    r = run_skill(names[tool], args, out_dir=exp.res / "runs" / f"{model.replace('/', '_')}_{t['id']}_{len(outs)}")
                    from plugins.llm_assistant.privacy import sanitize_result
                    o = sanitize_result(r, cloud=False)
                except Exception as exc:  # noqa: BLE001
                    o = f"ERROR: {exc}"
                outs.append(o)
                return o
            t0 = time.time()
            try:
                res = prov.run(SYSTEM, t["question"], specs, real, max_steps=6)
                chk = guardrails.check(res.text, outs, t["question"])
                grows.append({"model": model, "id": t["id"], "n_tool_calls": len(outs), "n_numbers": chk["n_numbers_checked"],
                              "n_unverified": len(chk["unverified"]), "any_unverified": bool(chk["unverified"]),
                              "seconds": time.time() - t0})
            except Exception as exc:  # noqa: BLE001
                grows.append({"model": model, "id": t["id"], "error": str(exc)[:200]})
        print(model, "grounding done", flush=True)
    r = pd.DataFrame(rows)
    g = pd.DataFrame(grows)
    exp.table(r, "e12_routing_raw.csv")
    exp.table(g, "e12_grounding_raw.csv")
    summ = r.groupby("model", as_index=False).agg(skill_accuracy=("skill_correct", "mean"), params_accuracy=("params_correct", "mean"),
                                                   median_seconds=("seconds", "median"), n=("id", "size"))
    if len(g) and "any_unverified" in g:
        gs = g.groupby("model", as_index=False).agg(frac_answers_with_untraceable_number=("any_unverified", "mean"),
                                                    mean_numbers_per_answer=("n_numbers", "mean"))
        summ = summ.merge(gs, on="model", how="left")
    exp.table(summ, "e12_summary.csv")
    by_skill = r.groupby(["model", "gold"], as_index=False)["skill_correct"].mean()
    exp.table(by_skill, "e12_accuracy_by_skill.csv")
    exp.summary.update({"summary": summ.to_dict("records")})
    exp.note("Supplementary experiment: the core system does not depend on any LLM.")
    exp.note("Routing uses stubbed tool results so it measures tool choice, not analysis quality.")
    exp.finish([("Summary", md_table(summ)), ("Accuracy by gold skill", md_table(by_skill))])
    print(summ)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen/qwen3.5-9b,qwen/qwen3.6-27b")
    ap.add_argument("--endpoint", default="http://localhost:1234/v1")
    ap.add_argument("--n-ground", type=int, default=30)
    a = ap.parse_args()
    main(a.models.split(","), a.endpoint, a.n_ground)
