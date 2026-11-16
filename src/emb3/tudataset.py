"""Downloads TUDataset archives and parses them into a GraphDataset.

The archives are read in place, without extracting them. Format reference:
https://chrsmrrs.github.io/datasets/docs/format/
"""
from __future__ import annotations

import hashlib
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from emb3.graphs import GraphDataset, canonical_edges


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, raw_dir: Path, base_url: str, expected_sha256: Optional[str] = None) -> Tuple[Path, str]:
    """Fetches <base_url>/<name>.zip unless it is already there. Returns (path, sha256)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "{}.zip".format(name)
    if not target.exists():
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        partial = raw_dir / "{}.zip.part".format(name)
        url = "{}/{}.zip".format(base_url.rstrip("/"), name)
        with urllib.request.urlopen(url, timeout=120, context=context) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(target)
    digest = sha256_of(target)
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(
            "{} has SHA-256 {} but the configuration expects {}. Delete the file to download it "
            "again, or update the configuration if the archive really changed.".format(target, digest, expected_sha256)
        )
    return target, digest


def _read_ints(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with archive.open(member) as handle:
        frame = pd.read_csv(handle, header=None, skipinitialspace=True, dtype=np.int64)
    return frame.to_numpy()


def parse(zip_path: Path, name: str, label_kind: str) -> Tuple[GraphDataset, Dict[str, Any]]:
    """Returns the dataset and a record of what the edge clean-up removed."""
    prefix = "{0}/{0}_".format(name)
    with zipfile.ZipFile(zip_path) as archive:
        raw_edges = _read_ints(archive, prefix + "A.txt") - 1
        graph_of_node = _read_ints(archive, prefix + "graph_indicator.txt")[:, 0] - 1
        graph_labels = _read_ints(archive, prefix + "graph_labels.txt")[:, 0]
        raw_node_labels = _read_ints(archive, prefix + "node_labels.txt")[:, 0] if label_kind == "native" else None

    num_graphs, total_nodes = len(graph_labels), len(graph_of_node)
    if np.any(np.diff(graph_of_node) < 0) or graph_of_node[0] != 0 or graph_of_node[-1] != num_graphs - 1:
        raise ValueError("{}: nodes are not grouped by graph".format(name))
    num_nodes = np.bincount(graph_of_node, minlength=num_graphs)
    if np.any(num_nodes == 0):
        raise ValueError("{}: a graph has no nodes".format(name))
    if raw_edges.min() < 0 or raw_edges.max() >= total_nodes:
        raise ValueError("{}: an edge refers to a node that does not exist".format(name))

    edges = canonical_edges(raw_edges, total_nodes)
    proper = raw_edges[raw_edges[:, 0] != raw_edges[:, 1]]
    directed_unique = len(np.unique(proper[:, 0] * total_nodes + proper[:, 1]))
    cleaning = {
        "entries_in_file": int(len(raw_edges)),
        "self_loops_removed": int(len(raw_edges) - len(proper)),
        "duplicate_entries_removed": int(len(proper) - directed_unique),
        "edges_listed_in_one_direction_only": int(2 * len(edges) - directed_unique),
        "undirected_edges_kept": int(len(edges)),
    }

    class_values, y = np.unique(graph_labels, return_inverse=True)
    node_labels = None
    if raw_node_labels is not None:
        if len(raw_node_labels) != total_nodes:
            raise ValueError("{}: node label count does not match node count".format(name))
        node_labels = np.unique(raw_node_labels, return_inverse=True)[1]

    dataset = GraphDataset(
        name=name, num_nodes=num_nodes, edges=edges, y=y,
        label_kind=label_kind, node_labels=node_labels, class_values=class_values,
    )
    return dataset, cleaning
