#!/usr/bin/env bash
# Builds the two pinned environments under .venvs/ from the lock files in requirements/.
# Needs uv (https://docs.astral.sh/uv/). Re-running rebuilds both environments from scratch.
#
#   .venvs/karateclub  Python 3.10  Graph2Vec, FGSD
#   .venvs/main        Python 3.12  GIN, WL kernel, evaluation
set -euo pipefail
cd "$(dirname "$0")/.."

KC_PYTHON=3.10.21
MAIN_PYTHON=3.12.14

uv venv --clear --managed-python --python "$KC_PYTHON" .venvs/karateclub
uv pip install --python .venvs/karateclub -r requirements/karateclub.txt

# --torch-backend cpu selects the CPU-only PyTorch wheels on Linux; on macOS it changes nothing.
uv venv --clear --managed-python --python "$MAIN_PYTHON" .venvs/main
uv pip install --python .venvs/main --torch-backend cpu -r requirements/main.txt

echo "Environments ready: .venvs/karateclub and .venvs/main"
