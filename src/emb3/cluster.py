"""k-means clustering of the graph representations, scored against the class labels, for
one dataset, method and embedding dimension (main environment):

    scripts/run.sh main -m emb3.cluster --dataset ENZYMES --method graph2vec --dimension 128
    scripts/run.sh main -m emb3.cluster --dataset ENZYMES --method wl_kernel

Every method is clustered in its reference setting, the first classification candidate in
configs/default.yaml: clustering has no labels to tune on. k is the number of classes.
Two protocols:

  test_folds  For every cross-validation test fold, only that fold's test graphs are
              clustered. This is the protocol that includes GIN, whose embedding comes from
              the model trained on the fold's training graphs: it never saw these graphs'
              labels. Graph2Vec and FGSD use the embedding of seed r in repeat r, as in
              classification, so all four methods cluster exactly the same graphs.
  all_graphs  The whole dataset at once, for the label-free methods only (Graph2Vec, FGSD,
              WL kernel), once per seed 0 .. timing.runs - 1 (embedding seed and k-means seed).

Vectors are standardised over the graphs being clustered, then clustered with k-means
(scikit-learn, clustering.n_init restarts). The WL kernel has no vectors: it gets kernel
k-means on the normalised kernel, which is k-means in the WL feature space.

Scores: adjusted Rand index (ARI, 0 for a random partition, 1 for the classes exactly)
and normalised mutual information (NMI, arithmetic mean normalisation). NMI is not
corrected for chance, and on a small test fold its chance level is well above zero, so
each row also has `nmi_chance`: the NMI of the same clusters with their members shuffled
at random (mean over clustering.chance_permutations shuffles). `nmi_size` compares the
clusters with graph size instead of class: the graphs split into k groups of equal count
by number of nodes. It shows when k-means sorts graphs by size rather than by class.

It writes results/clustering/<dataset>/<method>[_d<dimension>].csv, one row per clustering.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from emb3.classify import METHODS, candidates
from emb3.config import load_config, project_path
from emb3.embed import embedding_path, kernel_path
from emb3.graphs import GraphDataset, load_dataset
from emb3.splits import load_splits
from emb3.train_gin import gin_path


def result_path(config: Dict[str, Any], dataset: str, method: str, dimension: Optional[int]) -> Path:
    name = method if dimension is None else "{}_d{}".format(method, dimension)
    return project_path(config, "results") / "clustering" / dataset / "{}.csv".format(name)


def reference(config: Dict[str, Any], method: str, dataset: str) -> Dict[str, Any]:
    """The method's reference setting: its first classification candidate."""
    return candidates(config, method, dataset)[0]


# --- clustering -------------------------------------------------------------------------

def kmeans(vectors: np.ndarray, k: int, seed: int, n_init: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(vectors.astype(np.float64))
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(scaled)


def _kernel_distances(kernel: np.ndarray, diagonal: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Squared feature-space distance of every graph to every cluster mean:
    k(i,i) - 2/|c| sum_{j in c} k(i,j) + 1/|c|^2 sum_{j,l in c} k(j,l). Empty clusters: inf."""
    n = len(labels)
    members = np.zeros((n, k), dtype=kernel.dtype)
    members[np.arange(n), labels] = 1
    sizes = members.sum(axis=0).astype(np.float64)
    cross = (kernel @ members).astype(np.float64)
    within = (members * cross).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        distances = diagonal[:, None] - 2 * cross / sizes + within / sizes ** 2
    distances[:, sizes == 0] = np.inf
    return distances


def _seed_clusters(kernel: np.ndarray, diagonal: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding in the kernel's feature space: centres are graphs, each drawn with
    probability proportional to its squared distance to the nearest centre so far."""
    n = len(diagonal)
    centres = [int(rng.integers(n))]
    nearest = np.maximum(diagonal + diagonal[centres[0]] - 2 * kernel[:, centres[0]], 0)
    for _ in range(1, k):
        total = nearest.sum()
        centre = int(rng.choice(n, p=nearest / total)) if total > 0 else int(rng.integers(n))
        centres.append(centre)
        nearest = np.minimum(nearest, np.maximum(diagonal + diagonal[centre] - 2 * kernel[:, centre], 0))
    to_centres = diagonal[:, None] + diagonal[centres][None, :] - 2 * kernel[:, centres]
    return to_centres.argmin(axis=1)


def kernel_kmeans(kernel: np.ndarray, k: int, seed: int, n_init: int, max_iter: int = 300) -> np.ndarray:
    """Lloyd's algorithm in the feature space of a kernel, with k-means++ seeding and n_init
    restarts; returns the labels of the restart with the lowest within-cluster sum of
    squares. A cluster that runs empty takes over the graph farthest from its own cluster,
    as in scikit-learn's k-means."""
    n = kernel.shape[0]
    diagonal = np.diag(kernel).astype(np.float64)
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        labels = _seed_clusters(kernel, diagonal, k, rng)
        for _ in range(max_iter):
            distances = _kernel_distances(kernel, diagonal, labels, k)
            updated = distances.argmin(axis=1)
            empty = np.flatnonzero(np.bincount(updated, minlength=k) == 0)
            if len(empty):
                own = distances[np.arange(n), updated]
                for cluster, graph in zip(empty, np.argsort(own)[::-1]):
                    updated[graph] = cluster
            if np.array_equal(updated, labels):
                break
            labels = updated
        inertia = float(_kernel_distances(kernel, diagonal, labels, k)[np.arange(n), labels].sum())
        if inertia < best_inertia:
            best_labels, best_inertia = labels.copy(), inertia
    return best_labels


# --- scores -----------------------------------------------------------------------------

def size_groups(num_nodes: np.ndarray, k: int) -> np.ndarray:
    """Splits graphs into k groups of (nearly) equal count by number of nodes: a label-free
    property to compare clusters with, on the same scale as k classes."""
    edges = np.quantile(num_nodes, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(edges, num_nodes, side="right")


def scores(y: np.ndarray, labels: np.ndarray, seed: int, permutations: int, sizes: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    rng = np.random.default_rng(seed)
    chance = [normalized_mutual_info_score(y, rng.permutation(labels)) for _ in range(permutations)]
    return {
        "ari": float(adjusted_rand_score(y, labels)),
        "nmi": float(normalized_mutual_info_score(y, labels)),
        "nmi_chance": float(np.mean(chance)),
        "nmi_size": float(normalized_mutual_info_score(sizes, labels)),
    }


# --- representations --------------------------------------------------------------------

class Representations:
    """Loads the stored representation of one method: embeddings by seed, GIN embeddings by
    fold, or the WL kernel. Each file is read once."""

    def __init__(self, config: Dict[str, Any], dataset: GraphDataset, method: str, dimension: Optional[int]) -> None:
        self.config, self.dataset, self.method, self.dimension = config, dataset, method, dimension
        self.setting = reference(config, method, dataset.name)
        self._cache: Dict[Tuple[Any, ...], np.ndarray] = {}

    @property
    def setting_name(self) -> str:
        return self.setting.get("setting", self.setting.get("embedding"))

    @property
    def is_kernel(self) -> bool:
        return self.method == "wl_kernel"

    @staticmethod
    def _require(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError("{} is missing. Run the embeddings and GIN stages first.".format(path))
        return path

    def _cached(self, key: Tuple[Any, ...], load: Any) -> np.ndarray:
        if key not in self._cache:
            self._cache = {key: load()}   # one representation in memory at a time
        return self._cache[key]

    def for_seed(self, seed: int) -> np.ndarray:
        name = self.dataset.name
        if self.is_kernel:
            path = self._require(kernel_path(self.config, name, self.setting["embedding"]))
            return self._cached(("kernel",), lambda: np.load(path, allow_pickle=False))
        path = self._require(embedding_path(self.config, name, self.method, self.setting["embedding"], self.dimension, seed))
        vectors = self._cached(("seed", seed), lambda: np.load(path, allow_pickle=False))
        if self.setting.get("normalise"):
            vectors = vectors / np.maximum(vectors.sum(axis=1, keepdims=True), 1.0)
        return vectors

    def for_fold(self, repeat: int, fold: int, seed: int, test: np.ndarray) -> np.ndarray:
        if self.method != "gin":
            return self.for_seed(seed)
        path = self._require(gin_path(self.config, self.dataset.name, self.setting["setting"], self.dimension, repeat, fold))

        def load() -> np.ndarray:
            stored = np.load(path, allow_pickle=False)
            if not np.array_equal(stored["test_index"], test):
                raise RuntimeError("stored GIN fold does not match the current splits: {}".format(path))
            return stored["embedding"]

        return self._cached(("fold", repeat, fold), load)

    def cluster(self, data: np.ndarray, graphs: np.ndarray, k: int, seed: int) -> np.ndarray:
        n_init = self.config["clustering"]["n_init"]
        whole = len(graphs) == len(data) and np.array_equal(graphs, np.arange(len(data)))
        if self.is_kernel:
            return kernel_kmeans(data if whole else data[np.ix_(graphs, graphs)], k, seed, n_init)
        return kmeans(data if whole else data[graphs], k, seed, n_init)


def run(config: Dict[str, Any], dataset_name: str, method: str, dimension: Optional[int]) -> List[Dict[str, Any]]:
    dataset = load_dataset(dataset_name, config)
    splits = load_splits(dataset_name, config)
    splits.check(dataset.y)
    spec = config["cross_validation"][config["datasets"][dataset_name]["size"]]
    source = Representations(config, dataset, method, dimension)
    k, y = dataset.num_classes, dataset.y
    sizes = size_groups(dataset.num_nodes, k)
    permutations = config["clustering"]["chance_permutations"]
    rows: List[Dict[str, Any]] = []

    def record(protocol: str, repeat: Any, fold: Any, seed: int, graphs: np.ndarray, data: np.ndarray) -> None:
        start = time.perf_counter()
        labels = source.cluster(data, graphs, k, seed)
        seconds = time.perf_counter() - start
        row = {
            "dataset": dataset_name, "method": method, "dimension": dimension if dimension is not None else "",
            "setting": source.setting_name, "protocol": protocol, "repeat": repeat, "fold": fold, "seed": seed,
            "graphs": len(graphs), "clusters": k, "seconds": round(seconds, 3),
        }
        row.update(scores(y[graphs], labels, seed, permutations, sizes[graphs]))
        rows.append(row)

    for repeat in range(spec["repeats"]):
        seed = config["seeds"]["model"] + repeat
        for fold in range(spec["folds"]):
            _, _, test = splits.fold(repeat, fold)
            record("test_folds", repeat, fold, seed, test, source.for_fold(repeat, fold, seed, test))
    if method != "gin":
        everything = np.arange(len(dataset))
        for run_index in range(config["timing"]["runs"]):
            seed = config["seeds"]["model"] + run_index
            record("all_graphs", "", "", seed, everything, source.for_seed(seed))
    return rows


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dimension", type=int, help="embedding dimension (not used by wl_kernel)")
    parser.add_argument("--force", action="store_true", help="recompute even if the result exists")
    args = parser.parse_args(argv)
    if (args.dimension is None) != (args.method == "wl_kernel"):
        parser.error("--dimension is required for every method except wl_kernel")

    config = load_config()
    target = result_path(config, args.dataset, args.method, args.dimension)
    if target.exists() and not args.force:
        print("{} exists, skipping (use --force to recompute)".format(target), file=sys.stderr)
        return
    rows = run(config, args.dataset, args.method, args.dimension)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    parts = []
    for protocol in ("test_folds", "all_graphs"):
        chosen = [row for row in rows if row["protocol"] == protocol]
        if chosen:
            parts.append("{} ({}): ARI {:.3f}, NMI {:.3f} (chance {:.3f}, with size {:.3f})".format(
                protocol, len(chosen), np.mean([r["ari"] for r in chosen]), np.mean([r["nmi"] for r in chosen]),
                np.mean([r["nmi_chance"] for r in chosen]), np.mean([r["nmi_size"] for r in chosen])))
    print("{} {}{}: {} -> {}".format(args.method, args.dataset, "" if args.dimension is None else " d={}".format(args.dimension),
                                      "; ".join(parts), target), file=sys.stderr)


if __name__ == "__main__":
    main()
