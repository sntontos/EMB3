"""Cross-validated graph classification for one dataset, method and embedding dimension
(main environment):

    scripts/run.sh main -m emb3.classify --dataset ENZYMES --method graph2vec --dimension 128
    scripts/run.sh main -m emb3.classify --dataset ENZYMES --method wl_kernel

Protocol (configs/default.yaml, section `classification`):

  * Every method has at most four candidate settings. On the small datasets the candidate
    with the best validation accuracy is chosen separately in every fold; the validation
    graphs come from that fold's training graphs, so the test fold plays no part. On the
    large dataset the first candidate is used as is (fixed hyperparameters).
  * Every classifier is fitted on the same training graphs and scored on the same test
    graphs. Nothing is refitted on training plus validation, so that the SVM methods and
    GIN, which needs the validation graphs to pick its epoch, see the same amount of data.
  * Graph2Vec, FGSD: an RBF SVM on the standardised embedding. WL kernel: an SVM on the
    precomputed kernel. GIN: its own predictions, stored by emb3.train_gin.

It writes results/classification/<dataset>/<method>[_d<dimension>].csv, one row per test
fold with accuracy, macro-F1 and macro one-vs-rest AUC.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from emb3.config import load_config, project_path
from emb3.embed import embedding_path, kernel_path
from emb3.graphs import GraphDataset, load_dataset
from emb3.splits import load_splits
from emb3.train_gin import gin_path

METHODS = ("graph2vec", "fgsd", "gin", "wl_kernel")


def candidates(config: Dict[str, Any], method: str, dataset: str) -> List[Dict[str, Any]]:
    """The settings model selection chooses from: all of them on a small dataset, only the
    first (the reference setting) on a large one."""
    listed = config["classification"][method]
    if len(listed) > config["cross_validation"]["small"]["max_grid_settings"]:
        raise ValueError("{} has more candidate settings than the protocol allows".format(method))
    return listed if config["datasets"][dataset]["size"] == "small" else listed[:1]


def result_path(config: Dict[str, Any], dataset: str, method: str, dimension: Optional[int]) -> Path:
    name = method if dimension is None else "{}_d{}".format(method, dimension)
    return project_path(config, "results") / "classification" / dataset / "{}.csv".format(name)


def scores(y_true: np.ndarray, predicted: np.ndarray, proba: np.ndarray, num_classes: int) -> Dict[str, float]:
    from sklearn.metrics import f1_score, roc_auc_score

    labels = np.arange(num_classes)
    return {
        "accuracy": float(np.mean(predicted == y_true)),
        "macro_f1": float(f1_score(y_true, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_auc": float(roc_auc_score(y_true, proba, labels=labels, multi_class="ovr", average="macro")),
    }


class SVMMethod:
    """Graph2Vec, FGSD and the WL kernel: an SVM on stored embeddings or a stored kernel."""

    def __init__(self, config: Dict[str, Any], dataset: GraphDataset, method: str, dimension: Optional[int]) -> None:
        self.config, self.dataset, self.method, self.dimension = config, dataset, method, dimension
        self._cache: Dict[Tuple[Any, ...], np.ndarray] = {}

    def _features(self, candidate: Dict[str, Any], seed: int) -> np.ndarray:
        key = (candidate["embedding"], bool(candidate.get("normalise")), seed)
        if key not in self._cache:
            name = self.dataset.name
            if self.method == "wl_kernel":
                path = kernel_path(self.config, name, candidate["embedding"])
            else:
                path = embedding_path(self.config, name, self.method, candidate["embedding"], self.dimension, seed)
            if not path.exists():
                raise FileNotFoundError("{} is missing. Run the embeddings stage first.".format(path))
            features = np.load(path, allow_pickle=False).astype(np.float64)
            if candidate.get("normalise"):
                # FGSD histograms as shares of the counted node pairs instead of raw counts.
                features = features / np.maximum(features.sum(axis=1, keepdims=True), 1.0)
            self._cache[key] = features
        return self._cache[key]

    def _model(self, candidate: Dict[str, Any], seed: int, probability: bool) -> Any:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        if self.method == "wl_kernel":
            return SVC(kernel="precomputed", C=candidate["C"], probability=probability, random_state=seed)
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", C=candidate["C"], probability=probability, random_state=seed))

    def _slice(self, features: np.ndarray, rows: np.ndarray, train: np.ndarray) -> np.ndarray:
        return features[np.ix_(rows, train)] if self.method == "wl_kernel" else features[rows]

    def validation_accuracy(self, candidate: Dict[str, Any], seed: int, repeat: int, fold: int, train: np.ndarray, validation: np.ndarray) -> float:
        features, y = self._features(candidate, seed), self.dataset.y
        model = self._model(candidate, seed, probability=False).fit(self._slice(features, train, train), y[train])
        return float(np.mean(model.predict(self._slice(features, validation, train)) == y[validation]))

    def test(self, candidate: Dict[str, Any], seed: int, repeat: int, fold: int, train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        features, y = self._features(candidate, seed), self.dataset.y
        model = self._model(candidate, seed, probability=True).fit(self._slice(features, train, train), y[train])
        rows = self._slice(features, test, train)
        return model.predict(rows), model.predict_proba(rows)


class GINMethod:
    """GIN: reads the predictions that emb3.train_gin stored for each fold and setting."""

    def __init__(self, config: Dict[str, Any], dataset: GraphDataset, method: str, dimension: Optional[int]) -> None:
        self.config, self.dataset, self.dimension = config, dataset, dimension

    def _stored(self, candidate: Dict[str, Any], repeat: int, fold: int) -> Any:
        path = gin_path(self.config, self.dataset.name, candidate["setting"], self.dimension, repeat, fold)
        if not path.exists():
            raise FileNotFoundError("{} is missing. Run the GIN stages first.".format(path))
        return np.load(path, allow_pickle=False)

    def validation_accuracy(self, candidate: Dict[str, Any], seed: int, repeat: int, fold: int, train: np.ndarray, validation: np.ndarray) -> float:
        stored = self._stored(candidate, repeat, fold)
        if not np.array_equal(stored["validation_index"], validation):
            raise RuntimeError("stored GIN fold does not match the current splits")
        return float(np.mean(stored["validation_proba"].argmax(axis=1) == self.dataset.y[validation]))

    def test(self, candidate: Dict[str, Any], seed: int, repeat: int, fold: int, train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        stored = self._stored(candidate, repeat, fold)
        if not np.array_equal(stored["test_index"], test):
            raise RuntimeError("stored GIN fold does not match the current splits")
        proba = stored["test_proba"].astype(np.float64)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba.argmax(axis=1), proba


def run(config: Dict[str, Any], dataset_name: str, method: str, dimension: Optional[int]) -> List[Dict[str, Any]]:
    dataset = load_dataset(dataset_name, config)
    splits = load_splits(dataset_name, config)
    splits.check(dataset.y)
    spec = config["cross_validation"][config["datasets"][dataset_name]["size"]]
    options = candidates(config, method, dataset_name)
    runner = (GINMethod if method == "gin" else SVMMethod)(config, dataset, method, dimension)

    rows = []
    for repeat in range(spec["repeats"]):
        seed = config["seeds"]["model"] + repeat
        for fold in range(spec["folds"]):
            train, validation, test = splits.fold(repeat, fold)
            if len(options) > 1:
                validation_scores = [runner.validation_accuracy(option, seed, repeat, fold, train, validation) for option in options]
                chosen = int(np.argmax(validation_scores))   # ties go to the candidate listed first
            else:
                validation_scores, chosen = [float("nan")], 0
            start = time.perf_counter()
            predicted, proba = runner.test(options[chosen], seed, repeat, fold, train, test)
            seconds = time.perf_counter() - start
            row = {
                "dataset": dataset_name, "method": method, "dimension": dimension if dimension is not None else "",
                "repeat": repeat, "fold": fold, "seed": seed,
                "selected": options[chosen]["name"], "validation_accuracy": validation_scores[chosen],
                "train_graphs": len(train), "test_graphs": len(test), "classifier_seconds": round(seconds, 4),
            }
            row.update(scores(dataset.y[test], predicted, proba, dataset.num_classes))
            for option, value in zip(options, validation_scores):
                row["validation:" + option["name"]] = value
            rows.append(row)
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
    mean = lambda key: float(np.mean([row[key] for row in rows]))
    print("{} {}{}: {} folds, accuracy {:.3f}, macro-F1 {:.3f}, macro AUC {:.3f} -> {}".format(
        args.method, args.dataset, "" if args.dimension is None else " d={}".format(args.dimension),
        len(rows), mean("accuracy"), mean("macro_f1"), mean("macro_auc"), target), file=sys.stderr)


if __name__ == "__main__":
    main()
