#!/usr/bin/env bash
# Downloads the three datasets, stores them in canonical form under data/processed/,
# writes results/datasets/statistics.{md,csv} and creates splits/ if they do not exist yet.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/scripts/run.sh" main -m emb3.prepare_data "$@"
