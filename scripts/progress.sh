#!/usr/bin/env bash
# Shows which experiment stages are done, what is running, and the estimated time left.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
scripts/run.sh main -W ignore -m emb3.progress
if [ -f logs/experiments.log ]; then
  echo
  echo "Last lines of logs/experiments.log:"
  tail -n 3 logs/experiments.log | cut -c1-200
fi
