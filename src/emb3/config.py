"""Loads the experiment configuration from configs/default.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Returns the configuration as a dictionary. Defaults to configs/default.yaml."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(config: Dict[str, Any], key: str) -> Path:
    """Resolves an entry of the `paths` section against the repository root."""
    return ROOT / config["paths"][key]
