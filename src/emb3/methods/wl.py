"""Weisfeiler-Lehman subtree kernel through GraKeL (main environment). The baseline.

The kernel matrix is computed once over all graphs of a dataset, without labels, as is
usual for graph kernels. The SVM then works on the rows and columns of each split.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from grakel.kernels import VertexHistogram, WeisfeilerLehman

from emb3.convert import to_grakel
from emb3.graphs import GraphDataset
from emb3.measure import Meter


def kernel_matrix(dataset: GraphDataset, n_iter: int, normalize: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Returns the (graphs x graphs) WL subtree kernel and a record of parameters and cost.

    n_iter is the number of WL refinement rounds (the subtree height). With normalize=True
    every graph has kernel value 1 with itself, so large graphs do not dominate.
    """
    with Meter() as convert:
        graphs = to_grakel(dataset)
    kernel = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram, normalize=normalize)
    with Meter() as generate:
        matrix = kernel.fit_transform(graphs)
    record = {
        "method": "wl_kernel",
        "params": {"n_iter": n_iter, "normalize": normalize, "base_graph_kernel": "VertexHistogram"},
        "convert": convert.as_dict(),
        "train": None,
        "generate": generate.as_dict(),
    }
    return matrix, record
