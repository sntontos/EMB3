"""Cross-validation splits, generated once and stored in splits/<dataset>.json.

Every method reads the same stored splits. They are not regenerated per run, because the
two environments ship different scikit-learn versions and might shuffle differently.

Each repeat is one stratified K-fold partition. For model selection, each fold also has a
stratified validation set taken from that fold's training graphs (Errica et al., ICLR 2020).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from emb3.config import load_config, project_path


def make_splits(y: np.ndarray, n_folds: int, seeds: List[int], validation_fraction: float) -> List[Dict[str, Any]]:
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

    y = np.asarray(y)
    placeholder = np.zeros(len(y))
    repeats = []
    for seed in seeds:
        test_fold = np.full(len(y), -1, dtype=np.int64)
        outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_, test) in enumerate(outer.split(placeholder, y)):
            test_fold[test] = fold
        validation = []
        for fold in range(n_folds):
            train = np.flatnonzero(test_fold != fold)
            inner = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=1000 * seed + fold)
            _, chosen = next(inner.split(train, y[train]))
            validation.append(np.sort(train[chosen]).tolist())
        repeats.append({"seed": int(seed), "test_fold": test_fold.tolist(), "validation": validation})
    return repeats


class Splits:
    """Read access to the stored splits of one dataset."""

    def __init__(self, record: Dict[str, Any]) -> None:
        self.record = record
        self.n_graphs = int(record["n_graphs"])
        self.n_folds = int(record["n_folds"])
        self.n_repeats = len(record["repeats"])

    def fold(self, repeat: int, fold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(train, validation, test) graph indices. Train excludes validation; refit on
        train + validation once the hyperparameters are chosen."""
        entry = self.record["repeats"][repeat]
        test_fold = np.asarray(entry["test_fold"])
        test = np.flatnonzero(test_fold == fold)
        validation = np.asarray(entry["validation"][fold], dtype=np.int64)
        in_train = test_fold != fold
        in_train[validation] = False
        return np.flatnonzero(in_train), validation, test

    def __iter__(self) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
        for repeat in range(self.n_repeats):
            for fold in range(self.n_folds):
                yield (repeat, fold) + self.fold(repeat, fold)

    def check(self, y: np.ndarray) -> None:
        """Raises if the splits are not a valid stratified partition of a dataset with labels y."""
        y = np.asarray(y)
        if len(y) != self.n_graphs:
            raise ValueError("splits cover {} graphs, dataset has {}".format(self.n_graphs, len(y)))
        class_sizes = np.bincount(y)
        for repeat in range(self.n_repeats):
            seen = np.zeros(self.n_graphs, dtype=np.int64)
            for fold in range(self.n_folds):
                train, validation, test = self.fold(repeat, fold)
                together = np.concatenate([train, validation, test])
                if len(np.unique(together)) != self.n_graphs or len(together) != self.n_graphs:
                    raise ValueError("repeat {} fold {}: train, validation and test overlap or miss graphs".format(repeat, fold))
                seen[test] += 1
                # Stratified: each class is spread over the folds as evenly as its size allows.
                share = np.bincount(y[test], minlength=len(class_sizes))
                if np.any(share < class_sizes // self.n_folds) or np.any(share > -(-class_sizes // self.n_folds)):
                    raise ValueError("repeat {} fold {}: test fold is not stratified".format(repeat, fold))
            if not np.all(seen == 1):
                raise ValueError("repeat {}: a graph is not tested exactly once".format(repeat))


def splits_path(name: str, config: Optional[Dict[str, Any]] = None) -> Path:
    return project_path(config or load_config(), "splits") / "{}.json".format(name)


def save_splits(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")


def load_splits(name: str, config: Optional[Dict[str, Any]] = None) -> Splits:
    path = splits_path(name, config)
    if not path.exists():
        raise FileNotFoundError("{} is missing. Run scripts/prepare_data.sh first.".format(path))
    return Splits(json.loads(path.read_text(encoding="utf-8")))
