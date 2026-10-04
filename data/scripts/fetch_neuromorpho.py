"""Download labelled neuron reconstructions (SWC, CNG-standardised) from NeuroMorpho.org.

Archive: "Allen Cell Types" (mouse visual cortex, one lab and one reconstruction
protocol, so class differences are not confounded by lab style). Classes:
'principal cell' vs 'interneuron' (from NeuroMorpho's cell_type field). A seeded
balanced sample of up to ``--per-class`` neurons per class is downloaded.
"""
import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://neuromorpho.org/api/neuron/select?q=archive:%22Allen%20Cell%20Types%22&size=500&page={p}"
SWC = "https://neuromorpho.org/dableFiles/allen%20cell%20types/CNG%20version/{name}.CNG.swc"


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "neurostias/0.1"}), timeout=120) as r:
        return json.load(r)


def main(out: Path, per_class: int, seed: int):
    out.mkdir(parents=True, exist_ok=True)
    rows, p = [], 0
    while True:
        d = get_json(API.format(p=p))
        for n in d["_embedded"]["neuronResources"]:
            rows.append({"neuron_name": n["neuron_name"], "neuron_id": n["neuron_id"], "species": n["species"],
                         "brain_region": ";".join(n["brain_region"]), "cell_type": ";".join(n["cell_type"]),
                         "domain": n.get("domain"), "archive": n["archive"]})
        p += 1
        if p >= d["page"]["totalPages"]:
            break
    meta = pd.DataFrame(rows)
    meta["cls"] = meta["cell_type"].map(lambda s: "principal" if "principal cell" in s else ("interneuron" if "interneuron" in s else "other"))
    meta.to_csv(out / "allen_cell_types_all_metadata.csv", index=False)
    rng = random.Random(seed)
    chosen = []
    for c in ("principal", "interneuron"):
        names = meta.loc[(meta.cls == c) & meta.domain.str.contains("Dendrites", na=False), "neuron_name"].tolist()
        rng.shuffle(names)
        chosen += names[:per_class]
    sel = meta[meta.neuron_name.isin(chosen)].copy()
    swc_dir = out / "swc"
    swc_dir.mkdir(exist_ok=True)
    ok = []
    for name in sel.neuron_name:
        dest = swc_dir / f"{name}.CNG.swc"
        if not dest.exists():
            try:
                with urllib.request.urlopen(SWC.format(name=urllib.parse.quote(name)), timeout=120) as r:
                    dest.write_bytes(r.read())
                time.sleep(0.2)
            except Exception as exc:  # noqa: BLE001
                print("failed", name, exc)
                continue
        ok.append(name)
    sel = sel[sel.neuron_name.isin(ok)]
    sel.to_csv(out / "selected_metadata.csv", index=False)
    print(sel.cls.value_counts().to_dict(), "of", len(meta), "neurons in archive")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--out", required=True)
    a.add_argument("--per-class", type=int, default=200)
    a.add_argument("--seed", type=int, default=0)
    x = a.parse_args()
    main(Path(x.out), x.per_class, x.seed)
