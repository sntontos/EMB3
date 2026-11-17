"""Loads the experiment configuration from configs/default.yaml."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def config_file(path: Optional[Union[str, Path]] = None) -> Path:
    """The configuration in use: the given path, else $EMB3_CONFIG, else configs/default.yaml."""
    return Path(path or os.environ.get("EMB3_CONFIG") or DEFAULT_CONFIG)


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Returns the configuration as a dictionary."""
    with open(config_file(path), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def is_timed_run() -> bool:
    """True when this process was started as a clean timing run (scripts/run_experiments.sh
    sets EMB3_TIMED=1 for jobs it runs one at a time). Only such runs enter the cost tables."""
    return os.environ.get("EMB3_TIMED") == "1"


def project_path(config: Dict[str, Any], key: str) -> Path:
    """Resolves an entry of the `paths` section against the repository root."""
    return ROOT / config["paths"][key]
