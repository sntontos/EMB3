"""Trains GIN on one cross-validation fold and stores its predictions and graph embeddings.
Run it in the main environment, one process per fold:

    scripts/run.sh main -m emb3.train_gin --dataset ENZYMES --dimension 128 --repeat 0 --fold 0

The model is fitted on the fold's training graphs; the validation graphs choose the epoch.
It writes data/gin/<variant>/<dataset>/<setting>/d<dimension>/r<repeat>_f<fold>.npz with

    validation_index, validation_proba   for model selection
    test_index, test_proba               for the reported metrics
    embedding                            the pooled representation of every graph in the dataset

and a .json file of the same name with the parameters, training curve, timings and memory.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from emb3.config import is_timed_run, load_config, project_path
from emb3.embed import installed_versions, is_done, method_settings


def gin_path(config: Dict[str, Any], dataset: str, setting: str, dimension: int, repeat: int, fold: int, variant: str = "original") -> Path:
    folder = project_path(config, "data") / "gin" / variant / dataset / setting / "d{}".format(dimension)
    return folder / "r{}_f{}.npz".format(repeat, fold)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dimension", required=True, type=int)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--setting", default=None, help="named hyperparameter setting (default: the first one in the configuration)")
    parser.add_argument("--force", action="store_true", help="retrain even if the result exists")
    args = parser.parse_args(argv)

    config = load_config()
    settings = method_settings(config, "gin")
    setting = args.setting or next(iter(settings))
    if setting not in settings:
        parser.error("unknown setting {!r} for gin; configured: {}".format(setting, ", ".join(settings)))
    target = gin_path(config, args.dataset, setting, args.dimension, args.repeat, args.fold)
    if not args.force and is_done(target.with_suffix(".json"), target):
        print("{} is done, skipping (use --force to retrain)".format(target), file=sys.stderr)
        return

    from emb3.convert import to_pyg
    from emb3.graphs import load_dataset
    from emb3.measure import Meter, machine_state, thread_settings
    from emb3.methods import gin
    from emb3.seeds import seed_everything
    from emb3.splits import load_splits

    dataset = load_dataset(args.dataset, config)
    train, validation, test = load_splits(args.dataset, config).fold(args.repeat, args.fold)
    budget = config["gin_training"][config["datasets"][args.dataset]["size"]]
    params = dict(budget, **(settings[setting] or {}))
    seed = config["seeds"]["model"] + args.repeat
    seed_everything(seed)

    with Meter() as total:
        with Meter() as convert:
            graphs = to_pyg(dataset)
        model, record = gin.fit(dataset, graphs, train, validation, args.dimension, seed, **params)
        validation_proba = model.predict_proba([graphs[i] for i in validation])
        test_proba = model.predict_proba([graphs[i] for i in test])
        with Meter() as generate:
            embedding = model.embed(graphs)
    if embedding.shape != (len(dataset), args.dimension) or not np.isfinite(embedding).all():
        raise RuntimeError("unexpected embedding: shape {}".format(embedding.shape))

    test_accuracy = float(np.mean(test_proba.argmax(axis=1) == dataset.y[test]))
    record.update(
        dataset=args.dataset,
        variant="original",
        setting=setting,
        dimension=args.dimension,
        repeat=args.repeat,
        fold=args.fold,
        seed=seed,
        timed=is_timed_run(),
        graphs=len(dataset),
        train_graphs=len(train),
        test_accuracy=test_accuracy,
        convert=convert.as_dict(),
        generate=generate.as_dict(),
        total=total.as_dict(),
        threads=thread_settings(),
        machine=machine_state(),
        versions=installed_versions(),
        recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        validation_index=validation, validation_proba=validation_proba.astype(np.float32),
        test_index=test, test_proba=test_proba.astype(np.float32),
        embedding=embedding,
    )
    target.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("gin/{} {} d={} repeat {} fold {}: {} epochs (best {}), train {:.1f}s, embed {:.1f}s, peak {:.2f} GB, "
          "validation {:.3f}, test {:.3f} -> {}".format(
              setting, args.dataset, args.dimension, args.repeat, args.fold, record["epochs_run"], record["best_epoch"],
              record["train"]["seconds"], generate.seconds, max(record["train"]["peak_rss_bytes"], generate.peak_rss_bytes) / 2**30,
              record["best_validation_accuracy"], test_accuracy, target), file=sys.stderr)


if __name__ == "__main__":
    main()
