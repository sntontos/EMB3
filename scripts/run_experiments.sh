#!/usr/bin/env bash
# Runs one stage of the experiments, or all of them.
#
#   scripts/run_experiments.sh embeddings     Graph2Vec, FGSD, WL kernels        timed, one job at a time
#   scripts/run_experiments.sh gin-timed      GIN folds whose time is reported   timed, one job at a time
#   scripts/run_experiments.sh gin            all other GIN folds                JOBS at a time (default 2), low priority
#   scripts/run_experiments.sh classify       model selection and metrics        one job at a time
#   scripts/run_experiments.sh summarize      tables and figure from what exists
#   scripts/run_experiments.sh all            every stage for the small datasets, then for the large one
#
# Options after the stage restrict it (see emb3.plan), for example
#   scripts/run_experiments.sh embeddings --datasets ENZYMES IMDB-MULTI
#
# Finished jobs are skipped, so a stage can be started again after an interruption. A job
# that fails is listed in logs/failed_jobs.txt and the run goes on. Before every job that
# runs one at a time, the script waits while the Mac is on battery, so that all timed runs
# happen on mains power and a drained battery cannot cut a run short.
# Watch the progress with scripts/progress.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

stage="${1:?usage: scripts/run_experiments.sh <embeddings|gin-timed|gin|classify|summarize|all> [plan options]}"
shift
jobs="${JOBS:-2}"
export FAILED_LOG="$ROOT/logs/failed_jobs.txt"

# Keep the Mac awake (not the display) for as long as this script runs.
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i -w $$ &
fi

on_battery() {
  command -v pmset >/dev/null 2>&1 && pmset -g ps | head -1 | grep -q "Battery Power"
}

wait_for_mains() {
  if on_battery; then
    echo "-- $(date '+%a %H:%M') on battery: waiting for the charger" >&2
    while on_battery; do sleep 60; done
    echo "-- $(date '+%a %H:%M') charger connected, continuing" >&2
  fi
}

run_stage() {
  local name="$1"; shift
  if [ "$name" = summarize ]; then
    scripts/run.sh main -W ignore -m emb3.summarize
    return
  fi
  local list
  list="$(mktemp)"
  if ! scripts/run.sh main -W ignore -m emb3.plan "$name" "$@" > "$list"; then
    echo "-- could not plan stage $name" >&2
    rm -f "$list"
    return
  fi
  local count
  count="$(grep -c . "$list" || true)"
  if [ "$name" = gin ]; then
    echo "== $name: $count jobs, $jobs at a time at low priority ($(date '+%a %H:%M'))" >&2
    wait_for_mains
    # One line is one command made of plain words, so xargs can pass it on word by word.
    xargs -P "$jobs" -L 1 sh -c 'nice -n 10 "$@" || { echo "$(date "+%Y-%m-%d %H:%M") $*" >> "$FAILED_LOG"; echo "-- FAILED: $*" >&2; }' _ < "$list"
  else
    echo "== $name: $count jobs, one at a time ($(date '+%a %H:%M'))" >&2
    while IFS= read -r command; do
      [ -n "$command" ] || continue
      wait_for_mains
      if ! sh -c "$command" < /dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M') $command" >> "$FAILED_LOG"
        echo "-- FAILED: $command" >&2
      fi
    done < "$list"
  fi
  rm -f "$list"
}

echo "== code $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet HEAD -- src scripts configs requirements 2>/dev/null || echo ' with uncommitted changes'), started $(date '+%a %Y-%m-%d %H:%M')" >&2
if [ "$stage" = all ]; then
  # Small datasets first: their results are ready within a few hours, before the long
  # REDDIT-MULTI-12K runs.
  for size in small large; do
    group="$(scripts/run.sh main -W ignore -m emb3.plan datasets --size "$size")"
    [ -n "$group" ] || continue
    echo "==== $size datasets: $group" >&2
    for name in embeddings gin-timed gin classify summarize; do
      # shellcheck disable=SC2086
      run_stage "$name" --datasets $group "$@"
    done
  done
else
  run_stage "$stage" "$@"
fi
echo "== finished $(date '+%a %Y-%m-%d %H:%M')" >&2
