#!/usr/bin/env bash
# One-command setup: creates .venv, installs NeuroSTIAS (+ optional extras), runs the doctor.
#   ./install.sh            core + all analysis extras
#   ./install.sh core       core only
#   ./install.sh full       core + extras + benchmark baselines + LLM plugin
set -euo pipefail
cd "$(dirname "$0")"
PROFILE="${1:-all}"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.11 python3.12 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
echo "Using $($PY --version) at $(command -v $PY)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip setuptools wheel
case "$PROFILE" in
  core) .venv/bin/pip install -q -e ".[dev]" ;;
  all)  .venv/bin/pip install -q -e ".[all]" ;;
  full) .venv/bin/pip install -q -e ".[all,benchmarks,llm]" ;;
  *) echo "unknown profile $PROFILE"; exit 1 ;;
esac
.venv/bin/neurostias doctor
echo
echo "Done. Activate with:  source .venv/bin/activate"
