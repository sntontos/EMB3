"""Graph2Vec through KarateClub (karateclub environment).

Unsupervised and transductive: one model is fitted on all graphs of a dataset, without
labels, and each graph's document vector is its embedding.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from karateclub import Graph2Vec

from emb3.convert import to_networkx
from emb3.graphs import GraphDataset
from emb3.measure import Meter

# KarateClub's defaults, written out so the record shows them. Two values are ours:
#   attributed=True  node labels come from the dataset (or the node degree), as the
#                    assignment asks, instead of KarateClub's own degree fallback.
#   workers=1        gensim is only repeatable with a single worker thread (default: 4).
DEFAULTS: Dict[str, Any] = {
    "wl_iterations": 2,
    "attributed": True,
    "workers": 1,
    "down_sampling": 0.0001,
    "epochs": 10,
    "learning_rate": 0.025,
    "min_count": 5,
}


def embed(dataset: GraphDataset, dimension: int, seed: int, **overrides: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Returns the (graphs x dimension) embedding and a record of parameters and cost."""
    params = dict(DEFAULTS, **overrides)
    with Meter() as convert:
        graphs = to_networkx(dataset)
    model = Graph2Vec(dimensions=dimension, seed=seed, **params)
    # fit() extracts the WL features and trains Doc2Vec. The embedding is the trained
    # document vectors, so reading it out afterwards costs nothing.
    with Meter() as train:
        model.fit(graphs)
    with Meter() as generate:
        embedding = np.asarray(model.get_embedding(), dtype=np.float32)
    record = {
        "method": "graph2vec",
        "params": dict(params, dimensions=dimension, seed=seed),
        "convert": convert.as_dict(),
        "train": train.as_dict(),
        "generate": generate.as_dict(),
    }
    return embedding, record
