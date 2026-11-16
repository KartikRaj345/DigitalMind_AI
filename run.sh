#!/usr/bin/env bash
# Runs DigitalMind: prepares the environment, loads the demo data, starts the app.
#
#   ./run.sh
#   ./run.sh --setup --reset --no-reindex
#   ./run.sh --evaluate
#   ./run.sh --test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SETUP=0
RESET=0
NO_DEMO=0
NO_REINDEX=0
SKIP_DOWNLOAD=0
EVALUATE=0
TEST=0
HOST=""
PORT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup) SETUP=1 ;;
    --reset) RESET=1 ;;
    --no-demo) NO_DEMO=1 ;;
    --no-reindex) NO_REINDEX=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --evaluate) EVALUATE=1 ;;
    --test) TEST=1 ;;
    --host) HOST="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then PY=python; fi

if [[ $SETUP -eq 1 || ! -d .venv ]]; then
  echo "Creating virtual environment ..." >&2
  "$PY" -m venv .venv
fi

PYBIN=".venv/bin/python"
[[ -x "$PYBIN" ]] || PYBIN=".venv/Scripts/python.exe"

echo "Installing Python dependencies ..." >&2
"$PYBIN" -m pip install --quiet --upgrade pip
"$PYBIN" -m pip install --quiet -r requirements.txt

if [[ ! -d frontend/node_modules ]]; then
  echo "Installing frontend dependencies ..." >&2
  (cd frontend && npm install --silent)
fi

if [[ $TEST -eq 1 ]]; then
  "$PYBIN" -m pytest -q
  exit $?
fi

if [[ $EVALUATE -eq 1 ]]; then
  "$PYBIN" -m backend.scripts.generate_dataset
  "$PYBIN" -m backend.scripts.fetch_public_data
  "$PYBIN" -m backend.scripts.train_risk --model tfidf
  "$PYBIN" -m backend.scripts.evaluate_demo_scenarios
  "$PYBIN" -m backend.scripts.evaluate_extraction
  "$PYBIN" -m backend.scripts.evaluate_entity_resolution
  "$PYBIN" -m backend.scripts.evaluate_risk
  "$PYBIN" -m backend.scripts.evaluate_retrieval
  "$PYBIN" -m backend.scripts.error_analysis
  "$PYBIN" -m backend.scripts.make_report
  echo "Evaluation complete. See reports/evaluation_report.md" >&2
  exit 0
fi

ARGS=(-m backend.run)
[[ $RESET -eq 1 ]] && ARGS+=(--reset)
[[ $NO_DEMO -eq 1 ]] && ARGS+=(--no-demo)
[[ $NO_REINDEX -eq 1 ]] && ARGS+=(--no-reindex)
[[ $SKIP_DOWNLOAD -eq 1 ]] && ARGS+=(--skip-download)
[[ -n "$HOST" ]] && ARGS+=(--host "$HOST")
[[ -n "$PORT" ]] && ARGS+=(--port "$PORT")

exec "$PYBIN" "${ARGS[@]}"
