#!/usr/bin/env bash
# Records hardware and exact library versions for both environments in results/system/.
# Run it on the machine that produces the timings.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for env_name in karateclub main; do
  "$ROOT/scripts/run.sh" "$env_name" -m emb3.sysinfo --name "$env_name" --out "$ROOT/results/system"
done
