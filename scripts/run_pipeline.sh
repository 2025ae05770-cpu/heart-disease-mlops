#!/usr/bin/env bash
# Reproduce the whole ML pipeline locally (Linux / macOS / WSL).
#   ./scripts/run_pipeline.sh
#
# Uses a Linux-only virtual env (.venv-linux) so it never collides with a Windows .venv
# in the same folder (e.g. when running from /mnt/d/... inside WSL).
# Prefers `uv` (as in the Webinar 1 lab); it downloads Python 3.13 if the system lacks it.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-.venv-linux}"
PYTHON_VERSION="${PYTHON_VERSION:-3.14.3}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  if command -v uv >/dev/null 2>&1; then
    echo ">> Creating $VENV_DIR with uv (Python $PYTHON_VERSION)"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
  else
    PY="$(command -v "python$PYTHON_VERSION" || command -v python3.12 || command -v python3)"
    echo ">> Creating $VENV_DIR with $PY"
    "$PY" -m venv "$VENV_DIR" || {
      echo "ERROR: could not create a virtual env. Either install uv:"
      echo "         curl -LsSf https://astral.sh/uv/install.sh | sh"
      echo "       or the venv module:  sudo apt install python3-venv"
      exit 1
    }
  fi
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo ">> Using $(python --version) at $(command -v python)"

if command -v uv >/dev/null 2>&1; then
  uv pip install -r requirements.txt
else
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

export MLFLOW_DISABLE_AGENT_HINT=1
step() { echo; echo "=== $1 ==="; }

step "Lint";             flake8 src tests scripts dags
step "Unit tests";       pytest --cov=src --cov-report=term
step "Download + clean"; python scripts/download_data.py
step "EDA";              python -m src.eda.run_eda
step "Train + track";    python -m src.models.train
step "Batch inference";  python -m src.inference.predict

echo
echo "Done. Activate the env with:  source $VENV_DIR/bin/activate"
echo "MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"
echo "API:       uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
