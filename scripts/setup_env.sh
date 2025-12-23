#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/setup_env.sh [python_executable]
PY=${1:-python3}
VENV_DIR=.venv_311

echo "Setting up environment ($VENV_DIR) using $PY"
if [ ! -d "$VENV_DIR" ]; then
  $PY -m venv "$VENV_DIR"
else
  echo "$VENV_DIR already exists. Skipping venv creation."
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# Prefer lockfile when present
if [ -f requirements.lock ]; then
  echo "Found requirements.lock — installing pinned dependencies..."
  pip install -r requirements.lock
else
  echo "No requirements.lock found — installing top-level requirements and generating lockfile..."
  pip install -r requirements.txt
  pip freeze > requirements.lock
fi

echo "Done. Activate with: source $VENV_DIR/bin/activate"
