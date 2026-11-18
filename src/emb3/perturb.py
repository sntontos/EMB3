"""Perturbed copies of a dataset for the stability analysis (section 4.3c):

    scripts/run.sh main -m emb3.perturb --dataset ENZYMES

For every mode (add, remove) and fraction in the `stability` section of the configuration,
each graph gets that fraction of its edge count added as new random edges, or removed from
its existing edges. The number of edges per graph is fraction x edges, rounded up or down
at random in proportion to the remainder, so every graph is perturbed by the stated
fraction on average, small graphs included. Removed edges are drawn uniformly from the
graph's edges, added ones uniformly from the node pairs that are not yet connected (no
self-loops). Nodes are never added or removed, so an isolated node can appear.

Node labels: native labels are kept; degree labels (IMDB-MULTI, REDDIT-MULTI-12K) are
recomputed from the perturbed edges, since the label is defined as the degree.

Each copy is saved as data/processed/<variant>/<dataset>.npz with a .json record of the
edge counts; variants are named like add05 or remove20. Only numpy is used, so this module
runs in both environments.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from emb3.config import load_config
from emb3.graphs import GraphDataset, canonical_edges, load_dataset, processed_path

MODES = ("add", "remove")


def variant_name(mode: str, fraction: float) -> str:
    return "{}{:02d}".format(mode, int(round(fraction * 100)))


def variants(config: Dict[str, Any]) -> List[Tuple[str, str, float]]:
    """(name, mode, fraction) of every perturbed copy the configuration asks for."""
    spec = config["stability"]
    return [(variant_name(mode, fraction), mode, float(fraction)) for mode in spec["modes"] for fraction in spec["edge_fractions"]]


def _count(rng: np.random.Generator, expected: float) -> int:
    whole = int(np.floor(expected))
    return whole + int(rng.random() < expected - whole)


def _new_edges(rng: np.random.Generator, n: int, local: np.ndarray, count: int) -> np.ndarray:
    """`count` node pairs of an n-node graph that are not edges yet, drawn uniformly."""
    existing = set((local[:, 0] * n + local[:, 1]).tolist())
    free = n * (n - 1) // 2 - len(existing)
    count = min(count, free)
    if count <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    if n <= 300 or count > free // 2:
        # Small or nearly full graph: list every free pair and draw from them.
        low, high = np.triu_indices(n, k=1)
        keys = low * n + high
        keys = keys[~np.isin(keys, np.fromiter(existing, dtype=np.int64, count=len(existing)))]
        chosen = rng.choice(keys, size=count, replace=False)
    else:
        # Large sparse graph: draw pairs and reject the ones already taken.
        picked: set = set()
        while len(picked) < count:
            pairs = rng.integers(0, n, size=(2 * (count - len(picked)) + 8, 2))
            for a, b in pairs.tolist():
                if a == b:
                    continue
                key = min(a, b) * n + max(a, b)
                if key not in existing and key not in picked:
                    picked.add(key)
                    if len(picked) == count:
                        break
        chosen = np.fromiter(sorted(picked), dtype=np.int64, count=count)
    return np.stack([chosen // n, chosen % n], axis=1)


def perturb(dataset: GraphDataset, mode: str, fraction: float, seed: int) -> GraphDataset:
    if mode not in MODES:
        raise ValueError("mode must be one of {}".format(MODES))
    rng = np.random.default_rng([seed, MODES.index(mode), int(round(fraction * 1000))])
    blocks = []
    for index in range(len(dataset)):
        start = int(dataset.node_offsets[index])
        n = int(dataset.num_nodes[index])
        edges = dataset.edges[dataset.edge_offsets[index]:dataset.edge_offsets[index + 1]]
        count = _count(rng, fraction * len(edges))
        if mode == "remove":
            keep = np.ones(len(edges), dtype=bool)
            keep[rng.choice(len(edges), size=min(count, len(edges)), replace=False)] = False
            blocks.append(edges[keep])
        else:
            blocks.append(edges)
            blocks.append(_new_edges(rng, n, edges - start, count) + start)
    edges = canonical_edges(np.concatenate(blocks) if blocks else np.zeros((0, 2), dtype=np.int64), dataset.total_nodes)
    return GraphDataset(
        name=dataset.name, num_nodes=dataset.num_nodes, edges=edges, y=dataset.y, label_kind=dataset.label_kind,
        node_labels=dataset.node_labels if dataset.label_kind == "native" else None, class_values=dataset.class_values,
    )


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--force", action="store_true", help="recreate copies that exist")
    args = parser.parse_args(argv)

    config = load_config()
    original = load_dataset(args.dataset, config)
    seed = config["seeds"]["perturbation"]
    for name, mode, fraction in variants(config):
        target = processed_path(args.dataset, config, name)
        if target.exists() and not args.force:
            print("{} exists, skipping (use --force to recreate)".format(target), file=sys.stderr)
            continue
        copy = perturb(original, mode, fraction, seed)
        copy.save(target)
        before, after = len(original.edges), len(copy.edges)
        record = {"dataset": args.dataset, "variant": name, "mode": mode, "fraction": fraction, "seed": seed,
                  "edges_before": before, "edges_after": after, "change": (after - before) / before}
        target.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("perturb {} {}: {} -> {} edges ({:+.1%}) -> {}".format(args.dataset, name, before, after, record["change"], target), file=sys.stderr)


if __name__ == "__main__":
    main()
