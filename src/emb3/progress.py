"""Shows how far the experiments have got and estimates the time left.

    scripts/progress.sh

Estimates come from the job records already written: the median duration of earlier jobs
of the same kind, scaled by the embedding dimension when only another dimension has run.
They are rough until every kind of job has run at least once.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from emb3 import cluster, cluster_ablation, visualize
from emb3.classify import result_path
from emb3.config import ROOT, load_config, project_path
from emb3.embed import embedding_path, is_done, kernel_path, kernel_record_path
from emb3.plan import STAGES, Job, datasets_of_size, jobs
from emb3.train_gin import gin_path

OVERHEAD_SECONDS = 6.0           # starting Python, importing the libraries, loading the dataset
CLASSIFY_GUESS = {"small": 20.0, "large": 300.0}   # measured: about 1 minute per REDDIT-MULTI-12K fold
CLUSTER_GUESS = {"small": 15.0, "large": 120.0}    # rough guesses, per method and dimension
VISUALIZE_GUESS = {"small": 90.0, "large": 1200.0}  # rough guesses, per dataset
ABLATION_GUESS = {"small": 60.0, "large": 600.0}    # rough guesses, per dataset
PARALLEL_STAGES = {"gin": 2}     # keep in step with JOBS in scripts/run_experiments.sh


def _record_and_result(config: Dict[str, Any], job: Job) -> Tuple[Path, Optional[Path]]:
    o = job.options
    if job.module == "emb3.embed" and o["method"] == "wl_kernel":
        return kernel_record_path(config, o["dataset"], o["setting"], o["run"]), kernel_path(config, o["dataset"], o["setting"]) if o["run"] == 0 else None
    if job.module == "emb3.embed":
        target = embedding_path(config, o["dataset"], o["method"], o["setting"], o["dimension"], o["seed"])
        return target.with_suffix(".json"), target
    target = gin_path(config, o["dataset"], o["setting"], o["dimension"], o["repeat"], o["fold"])
    return target.with_suffix(".json"), target


def job_done(config: Dict[str, Any], job: Job) -> bool:
    if job.module == "emb3.classify":
        return result_path(config, job.options["dataset"], job.options["method"], job.options.get("dimension")).exists()
    if job.module == "emb3.cluster":
        return cluster.result_path(config, job.options["dataset"], job.options["method"], job.options.get("dimension")).exists()
    if job.module == "emb3.visualize":
        return all(visualize.figure_path(config, job.options["dataset"], projection).exists() for projection in visualize.PROJECTIONS)
    if job.module == "emb3.cluster_ablation":
        return cluster_ablation.result_path(config, job.options["dataset"]).exists()
    record, result = _record_and_result(config, job)
    return is_done(record, result, require_timed=job.timed)


def _durations(config: Dict[str, Any]) -> Dict[Tuple[Any, ...], List[float]]:
    """Seconds per job kind, from every record written so far."""
    data = project_path(config, "data")
    found: Dict[Tuple[Any, ...], List[float]] = {}
    for path in list((data / "embeddings").rglob("*.json")) + list((data / "gin").rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            seconds = float(record["total"]["seconds"])
        except (ValueError, KeyError, TypeError):
            continue
        key = (record["dataset"], record["method"], record.get("setting"), record.get("dimension"))
        found.setdefault(key, []).append(seconds)
    return found


def estimate(config: Dict[str, Any], job: Job, durations: Dict[Tuple[Any, ...], List[float]]) -> Optional[float]:
    o = job.options
    size = config["datasets"][o["dataset"]]["size"]
    if job.module == "emb3.classify":
        return CLASSIFY_GUESS[size] if o["method"] != "gin" else 10.0
    if job.module == "emb3.cluster":
        return CLUSTER_GUESS[size]
    if job.module == "emb3.visualize":
        return VISUALIZE_GUESS[size]
    if job.module == "emb3.cluster_ablation":
        return ABLATION_GUESS[size]
    method = "gin" if job.module == "emb3.train_gin" else o["method"]
    dimension = o.get("dimension")
    exact = durations.get((o["dataset"], method, o["setting"], dimension))
    if exact:
        return float(np.median(exact)) + OVERHEAD_SECONDS
    # Same dataset and method, another setting or dimension: scale linearly by dimension.
    similar = [(key[3], np.median(values)) for key, values in durations.items()
               if key[0] == o["dataset"] and key[1] == method and (dimension is None or key[3])]
    if not similar:
        return None
    if dimension is None:
        return float(np.median([value for _, value in similar])) + OVERHEAD_SECONDS
    known, seconds = min(similar, key=lambda item: abs(np.log(item[0] / dimension)))
    return float(seconds) * dimension / known + OVERHEAD_SECONDS


def _elapsed_seconds(text: str) -> float:
    days, _, clock = text.rpartition("-")
    parts = [int(part) for part in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def running_jobs() -> List[Tuple[str, float]]:
    try:
        listing = subprocess.run(["ps", "-ax", "-o", "etime=,command="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    running = []
    for line in listing.splitlines():
        match = re.search(r"-m (emb3\.(?:embed|train_gin|classify|cluster|cluster_ablation|visualize)) (.*)$", line)
        if match and "/bin/python" in line:
            running.append(("{} {}".format(match.group(1)[5:], match.group(2)), _elapsed_seconds(line.split()[0])))
    return running


def _duration(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return "{} min".format(max(minutes, 1) if seconds > 0 else 0)
    return "{} h {:02d} min".format(minutes // 60, minutes % 60)


def _power() -> str:
    try:
        first = subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"
    match = re.search(r"'([^']+)'", first)
    return match.group(1) if match else first


def main() -> None:
    config = load_config()
    durations = _durations(config)
    running = running_jobs()
    rows, total_left, unknown = [], 0.0, 0
    for size in ("small", "large"):
        group = datasets_of_size(config, size)
        if not group:
            continue
        timed_folds = {job.command.replace("EMB3_TIMED=1 ", "") for job in jobs(config, "gin-timed", datasets=group)}
        for stage in STAGES:
            planned = jobs(config, stage, datasets=group)
            if stage == "gin":
                # Folds that gin-timed trains are skipped by this stage, so they are not counted twice.
                planned = [job for job in planned if job.command not in timed_folds]
            if not planned:
                continue
            left = [job for job in planned if not job_done(config, job)]
            seconds = 0.0
            for job in left:
                guess = estimate(config, job, durations)
                if guess is None:
                    unknown += 1
                else:
                    seconds += guess
            seconds /= PARALLEL_STAGES.get(stage, 1)
            total_left += seconds
            rows.append(("{} datasets / {}".format(size, stage), len(planned) - len(left), len(planned), seconds))
    for _, elapsed in running:
        total_left -= elapsed / max(len(running), 1)
    total_left = max(total_left, 0.0)

    print("{:<32} {:>11}   {}".format("Stage", "Done", "Left (estimate)"))
    for name, done, count, seconds in rows:
        print("{:<32} {:>5} / {:<5}   {}".format(name, done, count, "-" if done == count else _duration(seconds)))
    print()
    if running:
        for command, elapsed in running:
            print("Running: {} ({} so far)".format(command, _duration(elapsed)))
    else:
        print("Running: nothing")
    print("Power: {}".format(_power()))
    failed = ROOT / "logs" / "failed_jobs.txt"
    if failed.exists() and failed.read_text(encoding="utf-8").strip():
        print("Failed jobs: {} (listed in logs/failed_jobs.txt)".format(len(failed.read_text(encoding="utf-8").strip().splitlines())))
    finish = datetime.now() + timedelta(seconds=total_left)
    print("Estimated time left: {}, finishing around {}{}".format(
        _duration(total_left), finish.strftime("%a %H:%M"),
        "" if not unknown else " (plus {} jobs of a kind that has not run yet)".format(unknown)))


if __name__ == "__main__":
    main()
