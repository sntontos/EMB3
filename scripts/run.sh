#!/usr/bin/env bash
# Runs Python inside one of the two environments.
#
#   scripts/run.sh karateclub -m emb3.sysinfo --name karateclub
#   scripts/run.sh main path/to/script.py
#
# It puts src/ on PYTHONPATH, which is how both environments find the emb3 package. The
# package is not installed into them: an editable install relies on a .pth file, and
# macOS can mark files under .venvs/ as hidden, which makes Python 3.12 ignore them.
#
# It also fixes PYTHONHASHSEED, which can only be set before the interpreter starts. gensim
# (used by Graph2Vec) lists a fixed hash seed and a single worker as conditions for
# repeatable runs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

env_name="${1:?usage: scripts/run.sh <karateclub|main> <python arguments>}"
shift
python="$ROOT/.venvs/$env_name/bin/python"
if [ ! -x "$python" ]; then
  echo "No environment at .venvs/$env_name. Run scripts/setup_envs.sh first." >&2
  exit 1
fi

export PYTHONHASHSEED=0
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" "$@"
