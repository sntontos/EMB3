"""t-SNE and UMAP maps of the graph representations, for the qualitative part of the
clustering task (main environment), one dataset at a time:

    scripts/run.sh main -m emb3.visualize --dataset ENZYMES

Each method in its reference setting at clustering.projection_dimension:

  Graph2Vec, FGSD  the standardised seed-0 embedding of every graph;
  WL kernel        the kernel distance sqrt(2 - 2 k(i, j)) between graphs, which is the
                   distance in the WL feature space because the kernel is normalised, given
                   to both projections as a precomputed metric;
  GIN              every graph, embedded by the model of repeat 0, fold 0. That model was
                   trained on about 80% of these graphs, so its map flatters it; the fair
                   comparison is the test-fold clustering in emb3.cluster.

Coordinates are cached under data/projections/<dataset>/, so a figure can be redrawn
without recomputing them (--force recomputes). Figures go to
results/figures/<projection>_<dataset>.png: one row per method and one column per class.
Each panel draws every graph in grey and one class in colour, which shows whether that
class gathers in one region; six or eleven classes in a single panel would need more
colours than can be told apart.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from emb3.classify import METHODS
from emb3.cluster import Representations
from emb3.config import load_config, project_path
from emb3.graphs import GraphDataset, load_dataset
from emb3.splits import load_splits

PROJECTIONS = {"tsne": "t-SNE", "umap": "UMAP"}
HIGHLIGHT, CONTEXT = "#2a78d6", "#cfcec7"   # categorical slot 1; a light neutral for the other graphs


def projection_path(config: Dict[str, Any], dataset: str, method: str, dimension: int, projection: str) -> Path:
    name = method if method == "wl_kernel" else "{}_d{}".format(method, dimension)
    return project_path(config, "data") / "projections" / dataset / "{}_{}.npy".format(name, projection)


def figure_path(config: Dict[str, Any], dataset: str, projection: str) -> Path:
    return project_path(config, "results") / "figures" / "{}_{}.png".format(projection, dataset)


def projection_input(config: Dict[str, Any], dataset: GraphDataset, method: str, dimension: int) -> Tuple[np.ndarray, str]:
    """What a projection is computed from: standardised vectors with the Euclidean metric, or
    for the WL kernel a matrix of feature-space distances with metric "precomputed"."""
    source = Representations(config, dataset, method, dimension)
    seed = config["seeds"]["model"]
    if method == "wl_kernel":
        kernel = source.for_seed(seed)
        distances = np.sqrt(np.clip(2.0 - 2.0 * kernel.astype(np.float32), 0.0, None))
        np.fill_diagonal(distances, 0.0)
        return distances, "precomputed"
    if method == "gin":
        _, _, test = load_splits(dataset.name, config).fold(0, 0)
        vectors = source.for_fold(0, 0, seed, test)
    else:
        vectors = source.for_seed(seed)
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(vectors.astype(np.float64)), "euclidean"


def project(data: np.ndarray, metric: str, projection: str, config: Dict[str, Any]) -> np.ndarray:
    seed = config["seeds"]["model"]
    options = config["clustering"][projection]
    if projection == "tsne":
        from sklearn.manifold import TSNE

        # PCA initialisation needs vectors; with precomputed distances it has to be random.
        init = "random" if metric == "precomputed" else "pca"
        return TSNE(n_components=2, metric=metric, init=init, random_state=seed, **options).fit_transform(data)
    import umap

    # A fixed random_state makes umap-learn run single-threaded, as every method here does.
    return umap.UMAP(n_components=2, metric=metric, random_state=seed, n_jobs=1, **options).fit_transform(data)


def draw(dataset: GraphDataset, projection: str, dimension: int, maps: Dict[str, np.ndarray], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from emb3.summarize import GRID, INK, INK_MUTED, INK_SECONDARY, METHOD_NAMES, SURFACE

    methods = [method for method in METHODS if method in maps]
    classes = np.arange(dataset.num_classes)
    size = 1.5
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9})
    figure, axes = plt.subplots(len(methods), len(classes), figsize=(1.3 + size * len(classes), 1.1 + size * len(methods)),
                                squeeze=False, facecolor=SURFACE)
    point = float(np.clip(2400 / len(dataset), 0.6, 7))
    for row, method in enumerate(methods):
        xy = maps[method]
        for column, label in enumerate(classes):
            axis = axes[row][column]
            mine = dataset.y == label
            axis.scatter(xy[~mine, 0], xy[~mine, 1], s=point, c=CONTEXT, linewidths=0, rasterized=True)
            # Semi-transparent, so that graphs drawn on the same spot (identical graphs are
            # common in IMDB-MULTI) build up colour and the grey ones still show through.
            axis.scatter(xy[mine, 0], xy[mine, 1], s=point * 1.5, c=HIGHLIGHT, alpha=0.45, linewidths=0, rasterized=True)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor(SURFACE)
            for spine in axis.spines.values():
                spine.set_color(GRID)
            if row == 0:
                axis.set_title("Class {}\n{} graphs".format(dataset.class_values[label], int(mine.sum())),
                               fontsize=8.5, color=INK_SECONDARY, pad=5)
            if column == 0:
                name = METHOD_NAMES[method] + ("\n(fold-0 model)" if method == "gin" else "")
                axis.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9.5, color=INK, labelpad=8)
    figure.suptitle("{}: {} maps, one class highlighted per column".format(dataset.name, PROJECTIONS[projection]),
                    x=0.01, y=0.99, ha="left", va="top", fontsize=11, fontweight="bold", color=INK)
    import textwrap

    caption = textwrap.fill("Graph2Vec, FGSD and GIN at {} dimensions, WL kernel by kernel distance. GIN's model was "
                            "trained on about 80% of these graphs.".format(dimension), width=int(figure.get_figwidth() * 15))
    figure.text(0.01, 0.01, caption, fontsize=8, color=INK_MUTED, ha="left", va="bottom")
    bottom = (0.1 + 0.15 * (caption.count("\n") + 1)) / figure.get_figheight()
    figure.tight_layout(rect=(0, bottom, 1, 1), h_pad=0.4, w_pad=0.4)   # leaves room for the title itself
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dimension", type=int, help="embedding dimension (default: clustering.projection_dimension)")
    parser.add_argument("--force", action="store_true", help="recompute the coordinates even if they are cached")
    args = parser.parse_args(argv)

    config = load_config()
    dataset = load_dataset(args.dataset, config)
    dimension = args.dimension or config["clustering"]["projection_dimension"]
    maps: Dict[str, Dict[str, np.ndarray]] = {projection: {} for projection in PROJECTIONS}
    timings: List[str] = []
    for method in METHODS:
        paths = {projection: projection_path(config, args.dataset, method, dimension, projection) for projection in PROJECTIONS}
        missing = [projection for projection, path in paths.items() if args.force or not path.exists()]
        data: Any = None
        if missing:
            data, metric = projection_input(config, dataset, method, dimension)
        for projection, path in paths.items():
            if projection in missing:
                start = time.perf_counter()
                coordinates = project(data, metric, projection, config).astype(np.float32)
                timings.append("{} {} {:.0f}s".format(method, projection, time.perf_counter() - start))
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, coordinates)
            maps[projection][method] = np.load(path, allow_pickle=False)
        del data
    written = [draw(dataset, projection, dimension, maps[projection], figure_path(config, args.dataset, projection))
               for projection in PROJECTIONS]
    print("visualize {} d={}: {} -> {}".format(args.dataset, dimension, ", ".join(timings) or "coordinates cached",
                                               ", ".join(str(path) for path in written)), file=sys.stderr)


if __name__ == "__main__":
    main()
