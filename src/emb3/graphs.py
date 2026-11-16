"""The canonical in-memory form of a graph dataset.

Every method reads its input from this one representation (through emb3.convert), so all
of them see exactly the same graphs. Only numpy is needed, so it loads in both environments.

All graphs of a dataset are stored together: nodes are numbered globally, graph by graph,
and every undirected edge is stored once as (low, high).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from emb3.config import load_config, project_path

LABEL_KINDS = ("native", "degree")


def canonical_edges(pairs: np.ndarray, total_nodes: int) -> np.ndarray:
    """Simple undirected edges: self-loops and duplicates removed, each edge once as
    (low, high), sorted. Sorting by the low endpoint also groups the edges by graph."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    low, high = pairs.min(axis=1), pairs.max(axis=1)
    proper = low != high
    keys = np.unique(low[proper] * total_nodes + high[proper])
    return np.stack([keys // total_nodes, keys % total_nodes], axis=1)


class GraphDataset:
    """A set of labelled graphs with one integer label per node.

    label_kind "native" keeps the node labels shipped with the dataset, renumbered from 0.
    label_kind "degree" uses each node's degree, as the assignment asks for datasets
    without node labels; these labels are derived from the edges, never stored separately.
    """

    def __init__(
        self,
        name: str,
        num_nodes: np.ndarray,
        edges: np.ndarray,
        y: np.ndarray,
        label_kind: str,
        node_labels: Optional[np.ndarray] = None,
        class_values: Optional[np.ndarray] = None,
    ) -> None:
        if label_kind not in LABEL_KINDS:
            raise ValueError("label_kind must be one of {}".format(LABEL_KINDS))
        self.name = name
        self.label_kind = label_kind
        self.num_nodes = np.asarray(num_nodes, dtype=np.int64)
        self.node_offsets = np.concatenate([[0], np.cumsum(self.num_nodes)]).astype(np.int64)
        self.edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        self.y = np.asarray(y, dtype=np.int64)
        self.class_values = np.arange(self.num_classes) if class_values is None else np.asarray(class_values)

        total = self.total_nodes
        if len(self.edges) and not (
            np.all(self.edges[:, 0] < self.edges[:, 1])
            and np.all(np.diff(self.edges[:, 0] * total + self.edges[:, 1]) > 0)
        ):
            raise ValueError("edges must come from canonical_edges()")
        graph_of_node = self.graph_of_node()
        if np.any(graph_of_node[self.edges[:, 0]] != graph_of_node[self.edges[:, 1]]):
            raise ValueError("an edge connects two different graphs")
        # Edges are sorted by their low endpoint, so each graph owns one contiguous block.
        self.edge_offsets = np.searchsorted(self.edges[:, 0], self.node_offsets, side="left").astype(np.int64)

        if label_kind == "degree":
            self.node_labels = self.degrees()
        elif node_labels is None:
            raise ValueError("native node labels were not given")
        else:
            self.node_labels = np.asarray(node_labels, dtype=np.int64)
        if len(self.node_labels) != total or len(self.y) != len(self.num_nodes):
            raise ValueError("array lengths do not match the number of nodes or graphs")

    def __len__(self) -> int:
        return len(self.num_nodes)

    @property
    def total_nodes(self) -> int:
        return int(self.node_offsets[-1])

    @property
    def num_classes(self) -> int:
        return int(self.y.max()) + 1

    @property
    def num_node_labels(self) -> int:
        """Size of the node-label vocabulary (labels are 0 .. num_node_labels - 1)."""
        return int(self.node_labels.max()) + 1

    @property
    def num_edges(self) -> np.ndarray:
        """Undirected edges per graph."""
        return np.diff(self.edge_offsets)

    def graph_of_node(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.num_nodes), dtype=np.int64), self.num_nodes)

    def degrees(self) -> np.ndarray:
        return np.bincount(self.edges.ravel(), minlength=self.total_nodes).astype(np.int64)

    def graph(self, index: int) -> Tuple[int, np.ndarray, np.ndarray]:
        """One graph as (number of nodes, edges with local node ids 0..n-1, node labels)."""
        start, stop = self.node_offsets[index], self.node_offsets[index + 1]
        edges = self.edges[self.edge_offsets[index]:self.edge_offsets[index + 1]] - start
        return int(stop - start), edges, self.node_labels[start:stop]

    def save(self, path: Union[str, Path]) -> None:
        arrays: Dict[str, Any] = {
            "name": np.array(self.name),
            "label_kind": np.array(self.label_kind),
            "num_nodes": self.num_nodes,
            "edges": self.edges,
            "y": self.y,
            "class_values": self.class_values,
        }
        if self.label_kind == "native":
            arrays["node_labels"] = self.node_labels
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GraphDataset":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                name=str(data["name"]),
                num_nodes=data["num_nodes"],
                edges=data["edges"],
                y=data["y"],
                label_kind=str(data["label_kind"]),
                node_labels=data["node_labels"] if "node_labels" in data.files else None,
                class_values=data["class_values"],
            )


def processed_path(name: str, config: Optional[Dict[str, Any]] = None) -> Path:
    return project_path(config or load_config(), "data") / "processed" / "{}.npz".format(name)


def load_dataset(name: str, config: Optional[Dict[str, Any]] = None) -> GraphDataset:
    """Loads a dataset prepared by scripts/prepare_data.sh."""
    path = processed_path(name, config)
    if not path.exists():
        raise FileNotFoundError("{} is missing. Run scripts/prepare_data.sh first.".format(path))
    return GraphDataset.load(path)
