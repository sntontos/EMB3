"""Computes and stores one label-free representation of a whole dataset: a Graph2Vec or FGSD
embedding (karateclub environment) or a WL kernel matrix (main environment). One process
per job, so that its memory is measured on its own:

    scripts/run.sh karateclub -m emb3.embed --method graph2vec --dataset ENZYMES --dimension 128 --seed 0
    scripts/run.sh main -m emb3.embed --method wl_kernel --dataset ENZYMES --setting iter3

A method's hyperparameters are chosen by name with --setting; the names and their values
are listed under methods.<method>.settings in configs/default.yaml.

Embeddings go to data/embeddings/<variant>/<dataset>/<method>/<setting>/d<dimension>_s<seed>.npy,
kernels to .../wl_kernel/<setting>/kernel.npy. A .json file of the same name holds the
parameters, timings, memory and library versions. GIN is supervised and trained per fold by
emb3.train_gin instead.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from emb3.config import is_timed_run, load_config, project_path

EMBEDDINGS = ("graph2vec", "fgsd")
KERNELS = ("wl_kernel",)
VERSIONS_OF = ("karateclub", "gensim", "networkx", "grakel", "torch", "torch-geometric", "scikit-learn", "numpy", "scipy")


def installed_versions() -> Dict[str, str]:
    versions = {}
    for name in VERSIONS_OF:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def method_settings(config: Dict[str, Any], method: str) -> Dict[str, Dict[str, Any]]:
    """The named hyperparameter settings of a method. Each maps to the values that differ
    from the method's defaults; the first one listed is used when none is named."""
    return config["methods"][method].get("settings") or {"library": {}}


def embedding_path(config: Dict[str, Any], dataset: str, method: str, setting: str, dimension: int, seed: int, variant: str = "original") -> Path:
    """Where an embedding lives. `variant` names the version of the dataset: "original",
    or a perturbed copy in the stability analysis."""
    folder = project_path(config, "data") / "embeddings" / variant / dataset / method / setting
    return folder / "d{}_s{}.npy".format(dimension, seed)


def kernel_path(config: Dict[str, Any], dataset: str, setting: str, variant: str = "original") -> Path:
    return project_path(config, "data") / "embeddings" / variant / dataset / "wl_kernel" / setting / "kernel.npy"


def is_done(record_path: Path, result_path: Optional[Path] = None, require_timed: Optional[bool] = None) -> bool:
    """Whether a job can be skipped: its record (and result) exist, and if a clean timing run
    is required (by default: when this process is one), the stored record came from one too.
    Otherwise it is recomputed, so that cost tables only contain timings of jobs that ran alone."""
    if not record_path.exists() or (result_path is not None and not result_path.exists()):
        return False
    if require_timed is None:
        require_timed = is_timed_run()
    if not require_timed:
        return True
    return bool(json.loads(record_path.read_text(encoding="utf-8")).get("timed"))


def kernel_record_path(config: Dict[str, Any], dataset: str, setting: str, run: int, variant: str = "original") -> Path:
    """Run 0 stores the kernel and kernel.json; later timing runs store only their record."""
    target = kernel_path(config, dataset, setting, variant)
    return target.with_suffix(".json") if run == 0 else target.parent / "kernel_run{}.json".format(run)


def load_embedding(config: Dict[str, Any], dataset: str, method: str, setting: str, dimension: int, seed: int, variant: str = "original") -> np.ndarray:
    return np.load(embedding_path(config, dataset, method, setting, dimension, seed, variant), allow_pickle=False)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", required=True, choices=EMBEDDINGS + KERNELS)
    parser.add_argument("--dataset")
    parser.add_argument("--dimension", type=int, help="embedding dimension (not used by wl_kernel)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", type=int, default=0, help="wl_kernel only: repeat number of a timing run; runs after the first store the record but not the matrix")
    parser.add_argument("--setting", default=None, help="named hyperparameter setting (default: the first one in the configuration)")
    parser.add_argument("--force", action="store_true", help="recompute even if the result exists")
    parser.add_argument("--list-settings", action="store_true", help="print the method's setting names and exit")
    args = parser.parse_args(argv)

    config = load_config()
    settings = method_settings(config, args.method)
    if args.list_settings:
        print(" ".join(settings))
        return
    setting = args.setting or next(iter(settings))
    if setting not in settings:
        parser.error("unknown setting {!r} for {}; configured: {}".format(setting, args.method, ", ".join(settings)))
    is_kernel = args.method in KERNELS
    if args.dataset is None or (args.dimension is None and not is_kernel):
        parser.error("--dataset is required, and --dimension for embedding methods")
    if is_kernel:
        target = kernel_path(config, args.dataset, setting)
        record_path = kernel_record_path(config, args.dataset, setting, args.run)
    else:
        target = embedding_path(config, args.dataset, args.method, setting, args.dimension, args.seed)
        record_path = target.with_suffix(".json")
    save_result = not (is_kernel and args.run > 0)
    if not args.force and is_done(record_path, target if save_result else None):
        print("{} is done, skipping (use --force to recompute)".format(record_path), file=sys.stderr)
        return

    from emb3.graphs import load_dataset
    from emb3.measure import Meter, machine_state, thread_settings
    from emb3.seeds import seed_everything

    dataset = load_dataset(args.dataset, config)
    seed_everything(args.seed)
    overrides = settings[setting] or {}
    with Meter() as total:
        if is_kernel:
            from emb3.methods import wl

            result, record = wl.kernel_matrix(dataset, **overrides)
            expected = (len(dataset), len(dataset))
            result = result.astype(np.float32)
        else:
            method = importlib.import_module("emb3.methods.{}".format(args.method))
            result, record = method.embed(dataset, args.dimension, args.seed, **overrides)
            expected = (len(dataset), args.dimension)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError("unexpected result: shape {}, finite {}".format(result.shape, bool(np.isfinite(result).all())))

    record.update(
        dataset=args.dataset,
        variant="original",
        setting=setting,
        dimension=None if is_kernel else args.dimension,
        seed=None if is_kernel else args.seed,
        run=args.run if is_kernel else args.seed,
        timed=is_timed_run(),
        graphs=len(dataset),
        total=total.as_dict(),
        threads=thread_settings(),
        machine=machine_state(),
        versions=installed_versions(),
        recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if save_result:
        np.save(target, result)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    stages = [record[stage] for stage in ("train", "generate") if record.get(stage)]
    print("{}/{} {}{}: {:.1f}s, peak {:.2f} GB -> {}".format(
        args.method, setting, args.dataset,
        " run {}".format(args.run) if is_kernel else " d={} seed={}".format(args.dimension, args.seed),
        sum(stage["seconds"] for stage in stages),
        max(stage["peak_rss_bytes"] for stage in stages) / 2**30, target if save_result else record_path), file=sys.stderr)


if __name__ == "__main__":
    main()
