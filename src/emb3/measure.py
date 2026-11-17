"""Wall time, CPU time and peak memory of a block of code. Used by every method.

    with Meter() as meter:
        model.fit(graphs)
    record = meter.as_dict()

Memory is the resident set size (RSS) of the whole process, so it includes the interpreter,
the imported libraries and the loaded dataset. `rss_before_bytes` is that baseline; the
difference to `peak_rss_bytes` is what the measured block added. Run each measured job in
its own process so that one job's memory cannot hide in another's.
"""
from __future__ import annotations

import os
import re
import resource
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional

import psutil


def _lifetime_peak_rss() -> int:
    """Highest RSS the process has ever had, in bytes (ru_maxrss is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


class Meter:
    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self.seconds: Optional[float] = None
        self.cpu_seconds: Optional[float] = None
        self.rss_before_bytes: Optional[int] = None
        self.peak_rss_bytes: Optional[int] = None

    def __enter__(self) -> "Meter":
        self._process = psutil.Process()
        self._stop = threading.Event()
        self.rss_before_bytes = self._process.memory_info().rss
        self._sampled_peak = self.rss_before_bytes
        self._lifetime_before = _lifetime_peak_rss()
        self._sampler = threading.Thread(target=self._sample, daemon=True)
        self._sampler.start()
        self._cpu_start = time.process_time()
        self._start = time.perf_counter()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self._sampled_peak = max(self._sampled_peak, self._process.memory_info().rss)

    def __exit__(self, *exc: Any) -> None:
        self.seconds = time.perf_counter() - self._start
        self.cpu_seconds = time.process_time() - self._cpu_start
        self._stop.set()
        self._sampler.join()
        peak = max(self._sampled_peak, self._process.memory_info().rss)
        # The kernel's own high-water mark catches spikes that fall between two samples,
        # but it only tells us about this block if the block raised it.
        lifetime_after = _lifetime_peak_rss()
        if lifetime_after > self._lifetime_before:
            peak = max(peak, lifetime_after)
        self.peak_rss_bytes = peak

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seconds": self.seconds,
            "cpu_seconds": self.cpu_seconds,
            "rss_before_bytes": self.rss_before_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


def thread_budget() -> int:
    """Threads each library may use, set by scripts/run.sh through EMB3_THREADS."""
    return int(os.environ.get("EMB3_THREADS", "1"))


def thread_settings() -> Dict[str, Any]:
    """The thread counts actually in force, for the job record."""
    settings: Dict[str, Any] = {"budget": thread_budget()}
    try:
        from threadpoolctl import threadpool_info

        settings["pools"] = {
            "{}:{}".format(pool.get("user_api"), pool.get("internal_api")): pool.get("num_threads")
            for pool in threadpool_info()
        }
    except ImportError:
        pass
    if "torch" in sys.modules:
        settings["torch"] = sys.modules["torch"].get_num_threads()
    return settings


def _pmset(*arguments: str) -> str:
    try:
        return subprocess.run(["pmset", *arguments], capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def machine_state() -> Dict[str, Any]:
    """Power source, Low Power Mode and any thermal limit when a job ran (macOS only).

    Stored in every job record, so the report can state that all timed runs were made
    under the same conditions."""
    if sys.platform != "darwin":
        return {}
    state: Dict[str, Any] = {}
    match = re.search(r"drawing from '([^']+)'", _pmset("-g", "ps"))
    if match:
        state["power_source"] = match.group(1)
    match = re.search(r"^\s*lowpowermode\s+(\d)", _pmset("-g"), re.MULTILINE)
    if match:
        state["low_power_mode"] = match.group(1) == "1"
    limits = [line.strip() for line in _pmset("-g", "therm").splitlines() if "=" in line]
    state["thermal_limits"] = limits
    return state
