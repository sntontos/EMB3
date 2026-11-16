"""Records the hardware and software facts the report has to state (assignment section 0.3).

    python -m emb3.sysinfo --name main --out results/system

writes results/system/main.json and rebuilds results/system/SYSTEM.md from every JSON
file in that directory, so the Markdown always covers all recorded environments.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

KEY_PACKAGES = [
    "karateclub", "gensim", "networkx", "torch", "torch-geometric", "grakel",
    "numpy", "scipy", "scikit-learn", "pandas", "umap-learn", "matplotlib",
]


def _run(command: List[str]) -> str:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip()


def _sysctl(name: str) -> str:
    return _run(["sysctl", "-n", name])


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _int(text: str) -> Optional[int]:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def cpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"architecture": platform.machine(), "logical_cores": os.cpu_count()}
    system = platform.system()
    if system == "Darwin":
        info["model"] = _sysctl("machdep.cpu.brand_string")
        info["physical_cores"] = _int(_sysctl("hw.physicalcpu"))
        info["logical_cores"] = _int(_sysctl("hw.logicalcpu")) or info["logical_cores"]
        performance = _int(_sysctl("hw.perflevel0.physicalcpu"))
        efficiency = _int(_sysctl("hw.perflevel1.physicalcpu"))
        if performance is not None and efficiency is not None:
            info["performance_cores"] = performance
            info["efficiency_cores"] = efficiency
    elif system == "Linux":
        cpuinfo = _read("/proc/cpuinfo")
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        if match:
            info["model"] = match.group(1).strip()
        else:
            lscpu = re.search(r"^Model name:\s*(.+)$", _run(["lscpu"]), re.MULTILINE)
            info["model"] = lscpu.group(1).strip() if lscpu else ""
    else:
        info["model"] = platform.processor()
    try:
        import psutil

        info.setdefault("physical_cores", psutil.cpu_count(logical=False))
    except ImportError:
        pass
    return info


def simd_info() -> Dict[str, bool]:
    """Which of the instruction sets named by the assignment the CPU supports."""
    system = platform.system()
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if system == "Darwin":
        if arm:
            return {"NEON": _sysctl("hw.optional.neon") == "1", "AVX2": False, "AVX-512": False}
        features = (_sysctl("machdep.cpu.features") + " " + _sysctl("machdep.cpu.leaf7_features")).upper()
        return {"NEON": False, "AVX2": "AVX2" in features, "AVX-512": "AVX512F" in features}
    if system == "Linux":
        flags = set()
        for line in _read("/proc/cpuinfo").splitlines():
            if line.lower().startswith(("flags", "features")):
                flags.update(line.split(":", 1)[1].split())
        return {
            "NEON": arm and ("asimd" in flags or "neon" in flags),
            "AVX2": "avx2" in flags,
            "AVX-512": "avx512f" in flags,
        }
    return {}


def memory_bytes() -> Optional[int]:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        pass
    if platform.system() == "Darwin":
        return _int(_sysctl("hw.memsize"))
    match = re.search(r"^MemTotal:\s*(\d+) kB", _read("/proc/meminfo"), re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def disk_info() -> Dict[str, Any]:
    system = platform.system()
    if system == "Darwin":
        fields = {}
        for line in _run(["diskutil", "info", "/"]).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
        solid = fields.get("Solid State", "")
        return {
            "type": "SSD" if solid.startswith("Yes") else ("HDD" if solid.startswith("No") else "unknown"),
            "file_system": fields.get("File System Personality", ""),
            "protocol": fields.get("Protocol", ""),
        }
    if system == "Linux":
        kinds = set()
        for block in sorted(Path("/sys/block").glob("*")):
            if block.name.startswith(("loop", "ram", "zram", "dm-")):
                continue
            rotational = _read(str(block / "queue" / "rotational")).strip()
            if rotational in ("0", "1"):
                kinds.add("SSD" if rotational == "0" else "HDD")
        return {"type": "/".join(sorted(kinds)) if kinds else "unknown"}
    return {"type": "unknown"}


def os_info() -> Dict[str, Any]:
    system = platform.system()
    info: Dict[str, Any] = {"system": system, "kernel": platform.release(), "in_container": os.path.exists("/.dockerenv")}
    if system == "Darwin":
        info["name"] = "macOS " + platform.mac_ver()[0]
    elif system == "Linux":
        match = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', _read("/etc/os-release"), re.MULTILINE)
        info["name"] = match.group(1) if match else "Linux"
    else:
        info["name"] = platform.platform()
    return info


def numeric_backends() -> Dict[str, Any]:
    """BLAS, OpenMP and PyTorch threading. These decide how fast the linear algebra runs."""
    backends: Dict[str, Any] = {}
    blas_name = ""
    try:
        import numpy as np

        try:
            blas = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
            version = str(blas.get("version", ""))
            blas_name = str(blas.get("name", ""))
            if version and version != "unknown":
                blas_name += " " + version
        except TypeError:  # numpy < 1.25 has no mode argument; the thread pools below name it
            pass
    except ImportError:
        pass
    try:
        import torch

        backends["torch_threads"] = torch.get_num_threads()
        backends["torch_cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        pass
    try:
        # Only libraries that are already loaded are listed, so this comes after the imports.
        from threadpoolctl import threadpool_info

        backends["thread_pools"] = [
            {
                "library": Path(pool.get("filepath", "")).name,
                "api": pool.get("user_api"),
                "implementation": pool.get("internal_api"),
                "version": pool.get("version"),
                "threads": pool.get("num_threads"),
            }
            for pool in threadpool_info()
        ]
        if not blas_name:
            for pool in backends["thread_pools"]:
                if pool["api"] == "blas":
                    blas_name = "{} {}".format(pool["implementation"], pool["version"] or "").strip()
                    break
    except ImportError:
        pass
    backends["numpy_blas"] = blas_name or "unknown"
    return backends


def _pools_text(record: Dict[str, Any]) -> str:
    pools = record["numeric_backends"].get("thread_pools") or []
    return ", ".join("{}: {}".format(pool["implementation"], pool["threads"]) for pool in pools) or "-"


def collect(name: str) -> Dict[str, Any]:
    packages = {dist.metadata["Name"]: dist.version for dist in metadata.distributions() if dist.metadata["Name"]}
    return {
        "environment": name,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu": cpu_info(),
        "simd": simd_info(),
        "memory_bytes": memory_bytes(),
        "disk": disk_info(),
        "os": os_info(),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "numeric_backends": numeric_backends(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def render_markdown(records: List[Dict[str, Any]]) -> str:
    """One hardware table (from the first record) and one version table across environments."""
    first = records[0]
    cpu, disk, system = first["cpu"], first["disk"], first["os"]
    cores = "{} physical, {} logical".format(cpu.get("physical_cores"), cpu.get("logical_cores"))
    if "performance_cores" in cpu:
        cores += " ({} performance + {} efficiency)".format(cpu["performance_cores"], cpu["efficiency_cores"])
    simd = ", ".join("{}: {}".format(key, _yes_no(value)) for key, value in first["simd"].items())
    memory = first.get("memory_bytes")
    disk_text = disk.get("type", "unknown")
    extras = [value for value in (disk.get("protocol"), disk.get("file_system")) if value]
    if extras:
        disk_text += " ({})".format(", ".join(extras))

    lines = [
        "# System record",
        "",
        "Generated by `scripts/record_system.sh`. Do not edit by hand.",
        "",
        "## Hardware and operating system",
        "",
        "| Item | Value |",
        "| --- | --- |",
        "| CPU | {} ({}) |".format(cpu.get("model") or "unknown", cpu.get("architecture")),
        "| Cores | {} |".format(cores),
        "| SIMD | {} |".format(simd or "unknown"),
        "| RAM | {} |".format("{:.0f} GB".format(memory / 2**30) if memory else "unknown"),
        "| Disk | {} |".format(disk_text),
        "| Operating system | {} (kernel {}) |".format(system.get("name"), system.get("kernel")),
        "| Inside a container | {} |".format(_yes_no(system.get("in_container", False))),
        "",
        "## Environments",
        "",
        "| | " + " | ".join("`{}`".format(record["environment"]) for record in records) + " |",
        "| --- |" + " --- |" * len(records),
        "| Recorded (UTC) | " + " | ".join(record["recorded_at"] for record in records) + " |",
        "| Python | " + " | ".join(record["python"]["version"] for record in records) + " |",
        "| NumPy BLAS | " + " | ".join(record["numeric_backends"].get("numpy_blas", "") or "-" for record in records) + " |",
        "| Thread pools (default threads) | " + " | ".join(_pools_text(record) for record in records) + " |",
        "| PyTorch threads (default) | " + " | ".join(str(record["numeric_backends"].get("torch_threads", "-")) for record in records) + " |",
    ]
    lowered = [{key.lower().replace("_", "-"): value for key, value in record["packages"].items()} for record in records]
    for package in KEY_PACKAGES:
        versions = [packages.get(package, "-") for packages in lowered]
        if any(version != "-" for version in versions):
            lines.append("| {} | ".format(package) + " | ".join(versions) + " |")
    lines += ["", "The complete package list of each environment is in the JSON file next to this one.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="environment name, used as the file name")
    parser.add_argument("--out", default="results/system", help="output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "{}.json".format(args.name)
    target.write_text(json.dumps(collect(args.name), indent=2) + "\n", encoding="utf-8")

    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(out.glob("*.json"))]
    (out / "SYSTEM.md").write_text(render_markdown(records), encoding="utf-8")
    print("Wrote {} and {}".format(target, out / "SYSTEM.md"), file=sys.stderr)


if __name__ == "__main__":
    main()
