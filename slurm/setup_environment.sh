#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  if [[ "${MODULE_PURGE:-1}" == "1" ]]; then
    module purge
  fi
  for module_name in ${MODULES:-python}; do
    module load "$module_name"
  done
fi

VENV="${VENV:-$HOME/eft_venv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[setup] repo: $REPO_DIR"
echo "[setup] venv: $VENV"
echo "[setup] python: $("$PYTHON_BIN" --version)"

"$PYTHON_BIN" - <<'PY'
import sys

minimum = (3, 10)
if sys.version_info < minimum:
    version = ".".join(map(str, sys.version_info[:3]))
    required = ".".join(map(str, minimum))
    raise SystemExit(f"Python {required}+ is required, but {version} is active. Load a newer Python module before setup.")
PY

if [[ ! -d "$VENV" ]]; then
  echo "[setup] creating virtual environment"
  "$PYTHON_BIN" -m venv "$VENV"
else
  echo "[setup] reusing existing virtual environment"
fi

source "$VENV/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[setup] installed packages:"
python -m pip freeze

echo
echo "[setup] done. Submit jobs with:"
echo "sbatch --export=ALL,VENV=\"$VENV\" slurm/run_neural_training.sbatch"
