"""Label-free variants of the clustering preprocessing, all reported side by side (main
environment), one dataset at a time:

    scripts/run.sh main -m emb3.cluster_ablation --dataset ENZYMES

The main clustering (emb3.cluster) fixes one preprocessing per method, chosen before any
result was seen. This asks whether another label-free choice recovers the classes better:

  Graph2Vec, FGSD  standardised (the main setting) and cosine (every embedding scaled to unit
                   length), each also followed by PCA to clustering.ablation.pca_components
                   dimensions; FGSD also as shares of node pairs instead of counts, which
                   removes most of the graph-size information from its histograms;
  WL kernel        kernel k-means for each WL depth in clustering.ablation.wl_settings, and
                   on the small datasets spectral clustering with the kernel as affinity
                   (it needs an eigendecomposition of the whole kernel).

Every variant is reported and none is picked by its agreement with the labels: that would
tune the clustering on the answer. The silhouette score, computed without labels in the
space that was clustered, shows what a label-free choice would prefer. It is computed on a
random sample of clustering.ablation.silhouette_sample graphs when a dataset is larger.

Whole dataset, k = number of classes, embedding dimension clustering.ablation.dimension,
seeds 0 .. timing.runs - 1 (embedding seed and clustering seed). It writes
results/clustering/ablation_<dataset>.csv, one row per method, variant and seed.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from emb3.cluster import Representations, kernel_kmeans, size_groups
from emb3.config import load_config, project_path
from emb3.embed import kernel_path
from emb3.graphs import load_dataset


def result_path(config: Dict[str, Any], dataset: str) -> Path:
    return project_path(config, "results") / "clustering" / "ablation_{}.csv".format(dataset)


def vector_variants(raw: np.ndarray, method: str, components: int, seed: int) -> Dict[str, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, normalize

    raw = raw.astype(np.float64)
    bases = {"standardised": StandardScaler().fit_transform(raw), "cosine": normalize(raw)}
    if method == "fgsd":
        shares = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1.0)
        bases["shares, standardised"] = StandardScaler().fit_transform(shares)
    variants: Dict[str, np.ndarray] = {}
    for name, vectors in bases.items():
        variants[name] = vectors
        variants["{} + PCA {}".format(name, components)] = PCA(components, random_state=seed).fit_transform(vectors)
    return variants


def silhouette(labels: np.ndarray, seed: int, limit: int, vectors: Optional[np.ndarray] = None, kernel: Optional[np.ndarray] = None) -> float:
    from sklearn.metrics import silhouette_score

    n = len(labels)
    rows = np.arange(n) if n <= limit else np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))
    if len(np.unique(labels[rows])) < 2:
        return float("nan")
    if kernel is not None:
        distances = np.sqrt(np.clip(2.0 - 2.0 * kernel[np.ix_(rows, rows)].astype(np.float64), 0.0, None))
        np.fill_diagonal(distances, 0.0)
        return float(silhouette_score(distances, labels[rows], metric="precomputed"))
    return float(silhouette_score(vectors[rows], labels[rows]))


def run(config: Dict[str, Any], dataset_name: str) -> List[Dict[str, Any]]:
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    dataset = load_dataset(dataset_name, config)
    options = config["clustering"]["ablation"]
    n_init = config["clustering"]["n_init"]
    small = config["datasets"][dataset_name]["size"] == "small"
    k, y = dataset.num_classes, dataset.y
    sizes = size_groups(dataset.num_nodes, k)
    seeds = [config["seeds"]["model"] + i for i in range(config["timing"]["runs"])]
    rows: List[Dict[str, Any]] = []

    def record(method: str, variant: str, main: bool, seed: int, cluster: Callable[[], np.ndarray], spread: Callable[[np.ndarray], float]) -> None:
        start = time.perf_counter()
        labels = cluster()
        seconds = time.perf_counter() - start
        rows.append({
            "dataset": dataset_name, "method": method, "variant": variant, "main": main, "seed": seed,
            "graphs": len(labels), "clusters": k,
            "ari": float(adjusted_rand_score(y, labels)), "nmi": float(normalized_mutual_info_score(y, labels)),
            "nmi_size": float(normalized_mutual_info_score(sizes, labels)), "silhouette": spread(labels),
            "seconds": round(seconds, 3),
        })

    for method in ("graph2vec", "fgsd"):
        source = Representations(config, dataset, method, options["dimension"])
        for seed in seeds:
            for variant, vectors in vector_variants(source.for_seed(seed), method, options["pca_components"], seed).items():
                record(method, variant, variant == "standardised", seed,
                       lambda: KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(vectors),
                       lambda labels: silhouette(labels, seed, options["silhouette_sample"], vectors=vectors))

    main_setting = Representations(config, dataset, "wl_kernel", None).setting["embedding"]
    for setting in options["wl_settings"]:
        kernel = np.load(kernel_path(config, dataset_name, setting), allow_pickle=False)
        affinity = kernel.astype(np.float64) if small else None
        for seed in seeds:
            spread = lambda labels: silhouette(labels, seed, options["silhouette_sample"], kernel=kernel)
            record("wl_kernel", "kernel k-means, {}".format(setting), setting == main_setting, seed,
                   lambda: kernel_kmeans(kernel, k, seed, n_init), spread)
            if affinity is not None:
                record("wl_kernel", "spectral clustering, {}".format(setting), False, seed,
                       lambda: SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed).fit_predict(affinity), spread)
        del kernel, affinity
    return rows


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--force", action="store_true", help="recompute even if the result exists")
    args = parser.parse_args(argv)

    config = load_config()
    target = result_path(config, args.dataset)
    if target.exists() and not args.force:
        print("{} exists, skipping (use --force to recompute)".format(target), file=sys.stderr)
        return
    rows = run(config, args.dataset)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("cluster_ablation {}: {} clusterings, {:.0f}s -> {}".format(
        args.dataset, len(rows), sum(row["seconds"] for row in rows), target), file=sys.stderr)


if __name__ == "__main__":
    main()
