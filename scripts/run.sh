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
# repeatable runs. And it caps the threads of the numerical libraries (see below).
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

# One thread budget for every library, so that methods are timed under the same conditions.
# It comes from `threads:` in the configuration (configs/default.yaml, or the file named by
# EMB3_CONFIG); set EMB3_THREADS to override it.
threads="${EMB3_THREADS:-$(awk '/^threads:/ {print $2; exit}' "${EMB3_CONFIG:-$ROOT/configs/default.yaml}")}"
export EMB3_THREADS="${threads:-1}"
export OMP_NUM_THREADS="$EMB3_THREADS" OPENBLAS_NUM_THREADS="$EMB3_THREADS" \
       MKL_NUM_THREADS="$EMB3_THREADS" VECLIB_MAXIMUM_THREADS="$EMB3_THREADS"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" "$@"
