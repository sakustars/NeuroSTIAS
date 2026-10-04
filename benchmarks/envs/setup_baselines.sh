#!/usr/bin/env bash
# Recreate the isolated environments used for baseline methods (exact versions in *_freeze.txt).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3.11}"
[ -d baselines ] || "$PY" -m venv baselines
baselines/bin/pip install -q --upgrade pip
baselines/bin/pip install -q "scanpy[leiden]" squidpy torch pybanksy paste-bio celltypist liana POT GraphST scikit-misc
baselines/bin/pip install -q --no-deps SpaGCN   # its 'louvain' dependency does not build on Apple Silicon; not needed (k-means init)
[ -d brian2 ] || "$PY" -m venv brian2
brian2/bin/pip install -q --upgrade pip
brian2/bin/pip install -q "numpy<2" "brian2==2.7.1"
echo "baseline environments ready"
