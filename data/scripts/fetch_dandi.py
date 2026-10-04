"""Download all NWB files of a published DANDI dandiset version (default 000006, version 0.220126.1855:
mouse ALM extracellular recordings during a delayed-response task)."""
import argparse
import json
import urllib.request
from pathlib import Path

API = "https://api.dandiarchive.org/api/dandisets/{d}/versions/{v}/assets/?page_size=200"


def main(out: Path, dandiset: str, version: str):
    out.mkdir(parents=True, exist_ok=True)
    url = API.format(d=dandiset, v=version)
    assets = []
    while url:
        with urllib.request.urlopen(url, timeout=120) as r:
            d = json.load(r)
        assets += d["results"]
        url = d.get("next")
    manifest = []
    for a in assets:
        dest = out / a["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != a["size"]:
            urllib.request.urlretrieve(f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/", dest)
        manifest.append({"path": a["path"], "asset_id": a["asset_id"], "size": a["size"]})
    (out / "manifest.json").write_text(json.dumps({"dandiset": dandiset, "version": version, "assets": manifest}, indent=1))
    print(len(manifest), "files")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dandiset", default="000006")
    p.add_argument("--version", default="0.220126.1855")
    a = p.parse_args()
    main(Path(a.out), a.dandiset, a.version)
