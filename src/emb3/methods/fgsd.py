"""FGSD through KarateClub (karateclub environment).

FGSD has no training step and no randomness: each graph is embedded on its own, as a
histogram of spectral distances taken from the pseudo-inverse of its normalized Laplacian.
It uses the graph structure only and ignores node labels. The number of histogram bins is
the embedding dimension.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from karateclub import FGSD

from emb3.convert import to_networkx
from emb3.graphs import GraphDataset
from emb3.measure import Meter

# KarateClub's default histogram range. The bin count is set to the embedding dimension
# (KarateClub's default is 200 bins).
DEFAULTS: Dict[str, Any] = {"hist_range": 20}


class FGSDWithFallback(FGSD):
    """KarateClub's FGSD, unchanged except for graphs on which it crashes.

    KarateClub computes the pseudo-inverse with a singular value decomposition. On some
    REDDIT-MULTI-12K graphs LAPACK gives up ("SVD did not converge") and the whole run
    would be lost. For those graphs only, the same pseudo-inverse is computed from the
    eigendecomposition instead, which is valid because the Laplacian is symmetric. The
    positions of these graphs are kept in `fallbacks`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fallbacks: List[int] = []
        self._position = -1

    def _calculate_fgsd(self, graph: nx.Graph) -> np.ndarray:
        self._position += 1
        try:
            return super()._calculate_fgsd(graph)
        except np.linalg.LinAlgError:
            self.fallbacks.append(self._position)
            return self._from_eigendecomposition(graph)

    def _from_eigendecomposition(self, graph: nx.Graph) -> np.ndarray:
        # Same steps as KarateClub's _calculate_fgsd, with pinv(..., hermitian=True).
        laplacian = np.asarray(nx.normalized_laplacian_matrix(graph).todense())
        inverse = np.linalg.pinv(laplacian, hermitian=True)
        ones = np.ones(laplacian.shape[0])
        distances = np.outer(np.diag(inverse), ones) + np.outer(ones, np.diag(inverse)) - 2 * inverse
        histogram, _ = np.histogram(distances.flatten(), bins=self.hist_bins, range=self.hist_range)
        return histogram


def embed(dataset: GraphDataset, dimension: int, seed: int, **overrides: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Returns the (graphs x dimension) embedding and a record of parameters and cost.

    The embedding holds raw pair counts, which grow with the square of the graph size.
    Any rescaling is left to the classifier, where it is tuned like other preprocessing.
    """
    params = dict(DEFAULTS, **overrides)
    with Meter() as convert:
        graphs = to_networkx(dataset)
    model = FGSDWithFallback(hist_bins=dimension, seed=seed, **params)
    with Meter() as generate:
        model.fit(graphs)
        embedding = np.asarray(model.get_embedding(), dtype=np.float32)
    record = {
        "method": "fgsd",
        "params": dict(params, hist_bins=dimension, seed=seed),
        "svd_fallback_graphs": model.fallbacks,
        "convert": convert.as_dict(),
        "train": None,
        "generate": generate.as_dict(),
    }
    return embedding, record
