"""Stability of each representation under edge perturbation (section 4.3c), for one dataset
and method (main environment):

    scripts/run.sh main -m emb3.stability --dataset ENZYMES --method graph2vec

Every method in its reference setting (the first classification candidate) at
stability.dimension; the WL kernel has no dimension. Perturbed copies come from
emb3.perturb and their representations from the stability-embeddings and stability-gin
stages. Two measurements:

Neighbourhood overlap. For every graph, its k nearest other graphs (k =
stability.nearest_neighbours): Euclidean distance on the standardised embedding, or the
kernel distance sqrt(k(i,i) + k(j,j) - 2 k(i,j)) for the WL kernel; ties go to the lower
graph index. Two representations of the same graphs are compared by the Jaccard overlap
of these sets, |A and B| / |A or B|, averaged over graphs. Embedding vectors are never
compared directly, because separate runs are not aligned.

    reference    the original data embedded with the two seeds of seeds.stability_pair: the
                 change caused by randomness alone. FGSD and the WL kernel are
                 deterministic, so theirs is 1.
    <variant>    original against a perturbed copy (add05, remove20, ...), both embedded
                 with the first seed of the pair.

GIN has no embedding of the whole dataset that is independent of a fold: it uses the
model of repeat 0, fold 0, which embeds every graph, and its reference trains that fold
again with the second seed.

Accuracy. The protocol of emb3.classify on every cross-validation fold, on the original
data and on each perturbed copy (repeat r uses embedding seed r), with one restriction:
model selection only chooses among the candidates that share the reference
representation, since only that one is recomputed on the perturbed copies. They differ in
the SVM's C, and for FGSD in counts against shares; GIN keeps its reference setting. On
the large dataset the reference setting is fixed anyway. The original accuracy is
recomputed the same way, so the difference shows the effect of the perturbation alone.

Writes results/stability/<dataset>/<method>_overlap.csv and <method>_accuracy.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from emb3.classify import METHODS, GINMethod, SVMMethod, candidates
from emb3.config import load_config, project_path
from emb3.embed import embedding_path, kernel_path
from emb3.graphs import load_dataset
from emb3.perturb import variants
from emb3.splits import load_splits
from emb3.train_gin import gin_path


def result_paths(config: Dict[str, Any], dataset: str, method: str) -> Tuple[Path, Path]:
    folder = project_path(config, "results") / "stability" / dataset
    return folder / "{}_overlap.csv".format(method), folder / "{}_accuracy.csv".format(method)


def gin_seed_argument(config: Dict[str, Any], seed: int) -> Optional[int]:
    """The --seed a GIN run of repeat 0 needs: none when it is that repeat's usual seed."""
    return None if seed == config["seeds"]["model"] else seed


# --- nearest neighbours -----------------------------------------------------------------

def nearest(distance_rows: Callable[[np.ndarray], np.ndarray], n: int, k: int, chunk: int = 1024) -> np.ndarray:
    """(n, k) indices of every graph's k nearest other graphs; ties go to the lower index."""
    found = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        distances = distance_rows(rows)
        distances[np.arange(len(rows)), rows] = np.inf
        found[rows] = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return found


def vector_distances(vectors: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Squared Euclidean distances on the standardised vectors, row block by row block."""
    from sklearn.preprocessing import StandardScaler

    z = StandardScaler().fit_transform(vectors.astype(np.float64))
    squares = (z ** 2).sum(axis=1)
    return lambda rows: np.maximum(squares[rows, None] + squares[None, :] - 2.0 * z[rows] @ z.T, 0.0)


def kernel_distances(kernel: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Squared feature-space distances k(i,i) + k(j,j) - 2 k(i,j), row block by row block."""
    diagonal = np.diag(kernel).astype(np.float64)
    return lambda rows: np.maximum(diagonal[rows, None] + diagonal[None, :] - 2.0 * kernel[rows].astype(np.float64), 0.0)


def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-graph Jaccard overlap of two neighbour lists (each row holds k distinct graphs)."""
    shared = (a[:, :, None] == b[:, None, :]).sum(axis=(1, 2))
    return shared / (a.shape[1] + b.shape[1] - shared)


class Neighbourhoods:
    """The k nearest neighbours of every graph under one method's representation."""

    def __init__(self, config: Dict[str, Any], dataset: str, method: str) -> None:
        self.config, self.dataset, self.method = config, dataset, method
        self.setting = candidates(config, method, dataset)[0]
        self.dimension = None if method == "wl_kernel" else config["stability"]["dimension"]
        self.k = config["stability"]["nearest_neighbours"]

    def _representation(self, variant: str, seed: int) -> Callable[[np.ndarray], np.ndarray]:
        config, name = self.config, self.dataset
        if self.method == "wl_kernel":
            path = kernel_path(config, name, self.setting["embedding"], variant)
        elif self.method == "gin":
            path = gin_path(config, name, self.setting["setting"], self.dimension, 0, 0, variant, gin_seed_argument(config, seed))
        else:
            path = embedding_path(config, name, self.method, self.setting["embedding"], self.dimension, seed, variant)
        if not path.exists():
            raise FileNotFoundError("{} is missing. Run the stability-embeddings and stability-gin stages first.".format(path))
        if self.method == "wl_kernel":
            return kernel_distances(np.load(path, allow_pickle=False))
        vectors = np.load(path, allow_pickle=False)["embedding"] if self.method == "gin" else np.load(path, allow_pickle=False)
        if self.setting.get("normalise"):
            vectors = vectors / np.maximum(vectors.sum(axis=1, keepdims=True), 1.0)
        return vector_distances(vectors)

    def of(self, variant: str, seed: int, n: int) -> np.ndarray:
        return nearest(self._representation(variant, seed), n, self.k)


# --- the two measurements ---------------------------------------------------------------

def overlap_rows(config: Dict[str, Any], dataset_name: str, method: str) -> List[Dict[str, Any]]:
    n = len(load_dataset(dataset_name, config))
    hood = Neighbourhoods(config, dataset_name, method)
    first, second = config["seeds"]["stability_pair"]
    base = hood.of("original", first, n)
    comparisons = [("reference", "", 0.0, hood.of("original", second, n))]
    comparisons += [(name, mode, fraction, hood.of(name, first, n)) for name, mode, fraction in variants(config)]
    rows = []
    for name, mode, fraction, other in comparisons:
        values = jaccard(base, other)
        rows.append({
            "dataset": dataset_name, "method": method, "setting": hood.setting["name"],
            "dimension": hood.dimension if hood.dimension is not None else "", "comparison": name, "mode": mode,
            "fraction": fraction, "seeds": "{} vs {}".format(first, second) if name == "reference" else "{} vs {}".format(first, first),
            "neighbours": hood.k, "graphs": n, "overlap_mean": float(values.mean()), "overlap_std": float(values.std(ddof=1)),
            "overlap_median": float(np.median(values)), "identical_share": float(np.mean(values == 1.0)),
        })
    return rows


def same_representation(config: Dict[str, Any], method: str, dataset: str) -> List[Dict[str, Any]]:
    """The classification candidates that use the reference representation (the first
    candidate's embedding, kernel or GIN setting), so they need nothing recomputed."""
    options = candidates(config, method, dataset)
    key = "setting" if method == "gin" else "embedding"
    return [option for option in options if option[key] == options[0][key]]


def accuracy_rows(config: Dict[str, Any], dataset_name: str, method: str) -> List[Dict[str, Any]]:
    from sklearn.metrics import f1_score

    splits = load_splits(dataset_name, config)
    spec = config["cross_validation"][config["datasets"][dataset_name]["size"]]
    options = same_representation(config, method, dataset_name)
    dimension = None if method == "wl_kernel" else config["stability"]["dimension"]
    rows = []
    for variant in ["original"] + [name for name, _, _ in variants(config)]:
        dataset = load_dataset(dataset_name, config, variant)
        runner = (GINMethod if method == "gin" else SVMMethod)(config, dataset, method, dimension, variant)
        for repeat in range(spec["repeats"]):
            seed = config["seeds"]["model"] + repeat
            for fold in range(spec["folds"]):
                train, validation, test = splits.fold(repeat, fold)
                candidate = options[0]
                if len(options) > 1:
                    scores = [runner.validation_accuracy(option, seed, repeat, fold, train, validation) for option in options]
                    candidate = options[int(np.argmax(scores))]   # ties go to the candidate listed first
                predicted = runner.predict(candidate, seed, repeat, fold, train, test)
                truth = dataset.y[test]
                rows.append({
                    "dataset": dataset_name, "method": method, "setting": candidate["name"],
                    "dimension": dimension if dimension is not None else "", "variant": variant,
                    "repeat": repeat, "fold": fold, "seed": seed, "test_graphs": len(test),
                    "accuracy": float(np.mean(predicted == truth)),
                    "macro_f1": float(f1_score(truth, predicted, labels=np.arange(dataset.num_classes), average="macro", zero_division=0)),
                })
    return rows


def _write(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--force", action="store_true", help="recompute even if the results exist")
    args = parser.parse_args(argv)

    config = load_config()
    overlap_path, accuracy_path = result_paths(config, args.dataset, args.method)
    if overlap_path.exists() and accuracy_path.exists() and not args.force:
        print("{} exists, skipping (use --force to recompute)".format(overlap_path.parent / args.method), file=sys.stderr)
        return
    overlaps = overlap_rows(config, args.dataset, args.method)
    _write(overlap_path, overlaps)
    accuracies = accuracy_rows(config, args.dataset, args.method)
    _write(accuracy_path, accuracies)
    mean = {variant: np.mean([r["accuracy"] for r in accuracies if r["variant"] == variant]) for variant in dict.fromkeys(r["variant"] for r in accuracies)}
    print("stability {} {}: overlap {}; accuracy {}".format(
        args.method, args.dataset,
        ", ".join("{} {:.3f}".format(r["comparison"], r["overlap_mean"]) for r in overlaps),
        ", ".join("{} {:.3f}".format(variant, value) for variant, value in mean.items())), file=sys.stderr)


if __name__ == "__main__":
    main()
