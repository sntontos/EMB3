"""Lists the jobs of one experiment stage as shell commands, one per line.

    scripts/run.sh main -m emb3.plan embeddings
    scripts/run.sh main -m emb3.plan gin --datasets ENZYMES --dimensions 32
    scripts/run.sh main -m emb3.plan datasets --size small      (names only)

scripts/run_experiments.sh runs these lists. The stages, in order:

    embeddings   Graph2Vec and FGSD embeddings and WL kernels. Timed: every job is repeated
                 `timing.runs` times (seeds 0, 1, ...) and must run alone.
    gin-timed    The GIN folds whose training time is reported: the reference setting on
                 `timing.runs` folds (all folds of the large dataset). Must run alone.
    gin          Every GIN fold of every candidate setting. Not timed, may run in parallel.
    classify     Model selection and test metrics for every dataset, method and dimension.
    cluster      k-means scores for every dataset, method and dimension, then the t-SNE and
                 UMAP figures and the preprocessing ablation of each dataset. Not timed.

Jobs that are already done are skipped by the commands themselves, so a list can always
be run again after an interruption.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from emb3.classify import METHODS, candidates
from emb3.config import load_config

STAGES = ("embeddings", "gin-timed", "gin", "classify", "cluster")
TIMED_STAGES = ("embeddings", "gin-timed")


class Job:
    """One command of a stage. `options` are its command-line options, in order."""

    def __init__(self, stage: str, env_name: str, module: str, options: Dict[str, Any]) -> None:
        self.stage = stage
        self.env_name = env_name
        self.module = module
        self.options = options
        self.timed = stage in TIMED_STAGES

    @property
    def command(self) -> str:
        arguments = " ".join("--{} {}".format(key, value) for key, value in self.options.items())
        return "{}scripts/run.sh {} -W ignore -m {} {}".format("EMB3_TIMED=1 " if self.timed else "", self.env_name, self.module, arguments)


def _unique(values: List[Any]) -> List[Any]:
    return list(dict.fromkeys(values))


def datasets_of_size(config: Dict[str, Any], size: str) -> List[str]:
    return [name for name, spec in config["datasets"].items() if spec["size"] == size]


def jobs(
    config: Dict[str, Any],
    stage: str,
    datasets: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    dimensions: Optional[List[int]] = None,
    runs: Optional[int] = None,
) -> List[Job]:
    datasets = datasets or list(config["datasets"])
    methods = methods or list(METHODS)
    dimensions = dimensions or config["embedding_dimensions"]
    runs = runs or config["timing"]["runs"]
    planned: List[Job] = []

    def add(env_name: str, module: str, **options: Any) -> None:
        planned.append(Job(stage, env_name, module, options))

    for dataset in datasets:
        size = config["datasets"][dataset]["size"]
        spec = config["cross_validation"][size]
        if stage == "embeddings":
            if size == "small" and runs < spec["repeats"]:
                raise ValueError("cross-validation repeat r uses the embedding of seed r, so at least {} runs are needed".format(spec["repeats"]))
            for method in methods:
                if method == "gin":
                    continue
                for setting in _unique([c["embedding"] for c in candidates(config, method, dataset)]):
                    if method == "wl_kernel":
                        for run in range(runs):
                            add("main", "emb3.embed", method="wl_kernel", dataset=dataset, setting=setting, run=run)
                        continue
                    for dimension in dimensions:
                        for seed in range(runs):
                            add("karateclub", "emb3.embed", method=method, dataset=dataset, setting=setting, dimension=dimension, seed=seed)
        elif stage in ("gin-timed", "gin") and "gin" in methods:
            options = candidates(config, "gin", dataset)
            if stage == "gin-timed":
                # Small datasets: the first `runs` folds. Large dataset: all of its folds.
                cells = [(options[0], 0, fold) for fold in range(spec["folds"] if size == "large" else min(runs, spec["folds"]))]
            else:
                cells = [(option, repeat, fold) for option in options for repeat in range(spec["repeats"]) for fold in range(spec["folds"])]
            for dimension in dimensions:
                for option, repeat, fold in cells:
                    add("main", "emb3.train_gin", dataset=dataset, dimension=dimension, setting=option["setting"], repeat=repeat, fold=fold)
        elif stage in ("classify", "cluster"):
            module = "emb3.{}".format(stage)
            for method in methods:
                if method == "wl_kernel":
                    add("main", module, dataset=dataset, method="wl_kernel")
                    continue
                for dimension in dimensions:
                    add("main", module, dataset=dataset, method=method, dimension=dimension)
            if stage == "cluster":
                add("main", "emb3.visualize", dataset=dataset)
                add("main", "emb3.cluster_ablation", dataset=dataset)
    return planned


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=STAGES + ("datasets",))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--dimensions", nargs="+", type=int)
    parser.add_argument("--runs", type=int, help="timing repeats (default: timing.runs in the configuration)")
    parser.add_argument("--size", choices=("small", "large"), help="with the `datasets` stage: which datasets to name")
    args = parser.parse_args(argv)
    config = load_config()
    if args.stage == "datasets":
        print(" ".join(datasets_of_size(config, args.size) if args.size else list(config["datasets"])))
        return
    print("\n".join(job.command for job in jobs(config, args.stage, args.datasets, args.methods, args.dimensions, args.runs)))


if __name__ == "__main__":
    main()
