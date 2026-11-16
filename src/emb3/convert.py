"""Converts a GraphDataset into the input format of each library.

Each function imports its library only when called, because no environment has all three.
Pass `indices` to convert a subset; the order of the result follows `indices`.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from emb3.graphs import GraphDataset


def _indices(dataset: GraphDataset, indices: Optional[Iterable[int]]) -> Iterable[int]:
    return range(len(dataset)) if indices is None else indices


def to_networkx(dataset: GraphDataset, indices: Optional[Iterable[int]] = None) -> List["networkx.Graph"]:
    """NetworkX graphs for KarateClub: nodes 0..n-1, node label in the "feature" attribute.

    KarateClub adds a self-loop to every node of the graphs it is given, in place, so build
    a fresh list for each fit instead of reusing one.
    """
    import networkx as nx

    graphs = []
    for index in _indices(dataset, indices):
        n, edges, labels = dataset.graph(index)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(edges.tolist())
        nx.set_node_attributes(graph, dict(enumerate(labels.tolist())), "feature")
        graphs.append(graph)
    return graphs


def to_grakel(dataset: GraphDataset, indices: Optional[Iterable[int]] = None) -> List["grakel.Graph"]:
    """GraKeL graphs with node labels, for the WL subtree kernel."""
    from grakel import Graph

    graphs = []
    for index in _indices(dataset, indices):
        n, edges, labels = dataset.graph(index)
        adjacency = {node: [] for node in range(n)}
        for u, v in edges.tolist():
            adjacency[u].append(v)
            adjacency[v].append(u)
        graphs.append(Graph(adjacency, node_labels=dict(enumerate(labels.tolist()))))
    return graphs


def to_pyg(dataset: GraphDataset, indices: Optional[Iterable[int]] = None) -> List["torch_geometric.data.Data"]:
    """PyTorch Geometric graphs for GIN.

    `x` holds the integer node label (shape [n, 1]), not a one-hot vector: with degree labels
    on REDDIT-MULTI-12K a one-hot matrix would not fit in memory. The model looks the label
    up in an embedding table, which is the same as a linear layer applied to the one-hot.
    """
    import torch
    from torch_geometric.data import Data

    graphs = []
    for index in _indices(dataset, indices):
        n, edges, labels = dataset.graph(index)
        both_directions = np.concatenate([edges, edges[:, ::-1]], axis=0).T
        graphs.append(
            Data(
                x=torch.from_numpy(np.ascontiguousarray(labels)).long().view(-1, 1),
                edge_index=torch.from_numpy(np.ascontiguousarray(both_directions)).long(),
                y=torch.tensor([int(dataset.y[index])]),
                num_nodes=n,
            )
        )
    return graphs
