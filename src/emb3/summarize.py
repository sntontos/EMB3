"""Builds the result tables and the accuracy-versus-cost figure from whatever has been run
so far (main environment):

    scripts/run.sh main -m emb3.summarize

Classification: results/classification/summary.{csv,md}, from the per-fold files written
by emb3.classify. Cost: results/cost/{runs,summary}.csv and summary.md, from the job
records of clean timing runs only. Figure: results/figures/accuracy_vs_cost.png.
Clustering: results/clustering/summary.{csv,md}, from the files written by emb3.cluster
(its t-SNE and UMAP figures are drawn by emb3.visualize).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from emb3.classify import METHODS, candidates
from emb3.config import load_config, project_path
from emb3.tables import markdown_table

METHOD_NAMES = {"graph2vec": "Graph2Vec", "fgsd": "FGSD", "gin": "GIN", "wl_kernel": "WL kernel"}
METRICS = ("accuracy", "macro_f1", "macro_auc")


def ci95_half_width(values: pd.Series) -> float:
    """Half-width of the 95% confidence interval of the mean (Student t)."""
    from scipy import stats

    n = len(values)
    if n < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def classification_folds(config: Dict[str, Any]) -> pd.DataFrame:
    files = sorted((project_path(config, "results") / "classification").glob("*/*.csv"))
    frames = [pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def classification_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method, dimension), group in folds.groupby(["dataset", "method", "dimension"], dropna=False, sort=False):
        row: Dict[str, Any] = {"dataset": dataset, "method": method, "dimension": dimension, "folds": len(group)}
        for metric in METRICS:
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_std"] = group[metric].std(ddof=1)
        row["accuracy_ci95"] = ci95_half_width(group["accuracy"])
        chosen = group["selected"].value_counts()
        row["most_selected"] = chosen.index[0]
        row["most_selected_share"] = chosen.iloc[0] / len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def _stage(record: Dict[str, Any], name: str, field: str) -> float:
    return float((record.get(name) or {}).get(field) or 0.0)


def cost_runs(config: Dict[str, Any]) -> pd.DataFrame:
    """One row per clean timing run. `reference` marks the runs of each method's reference
    setting, the ones the cost summary is built from."""
    data = project_path(config, "data")
    rows = []
    for path in sorted(list((data / "embeddings").rglob("*.json")) + list((data / "gin").rglob("*.json"))):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("timed") or record["dataset"] not in config["datasets"]:
            continue
        method = record["method"]
        first = candidates(config, method, record["dataset"])[0]
        rows.append({
            "dataset": record["dataset"],
            "method": method,
            "setting": record["setting"],
            "dimension": record.get("dimension"),
            "run": record.get("fold", record.get("run")),
            "reference": record["setting"] == first.get("setting", first.get("embedding")),
            "train_seconds": _stage(record, "train", "seconds"),
            "generate_seconds": _stage(record, "generate", "seconds"),
            "cpu_seconds": _stage(record, "train", "cpu_seconds") + _stage(record, "generate", "cpu_seconds"),
            "convert_seconds": _stage(record, "convert", "seconds"),
            "peak_rss_gb": max(_stage(record, "train", "peak_rss_bytes"), _stage(record, "generate", "peak_rss_bytes")) / 2**30,
            "threads": record["threads"]["budget"],
            "power_source": (record.get("machine") or {}).get("power_source", ""),
            "low_power_mode": (record.get("machine") or {}).get("low_power_mode", ""),
        })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["total_seconds"] = frame["train_seconds"] + frame["generate_seconds"]
    return frame


def cost_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method, dimension), group in runs[runs["reference"]].groupby(["dataset", "method", "dimension"], dropna=False, sort=False):
        total = group["total_seconds"]
        rows.append({
            "dataset": dataset, "method": method, "dimension": dimension, "setting": group["setting"].iloc[0], "runs": len(group),
            "total_seconds_median": total.median(), "total_seconds_mean": total.mean(),
            "total_seconds_std": total.std(ddof=1), "total_seconds_ci95": ci95_half_width(total),
            "train_seconds_median": group["train_seconds"].median(),
            "generate_seconds_median": group["generate_seconds"].median(),
            "peak_rss_gb_median": group["peak_rss_gb"].median(), "peak_rss_gb_max": group["peak_rss_gb"].max(),
        })
    return pd.DataFrame(rows)


def _dimension(value: Any) -> str:
    return "-" if pd.isna(value) or value == "" else str(int(value))


def _ordered(frame: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    frame = frame.copy()
    frame["_dataset"] = frame["dataset"].map({name: i for i, name in enumerate(config["datasets"])})
    frame["_method"] = frame["method"].map({name: i for i, name in enumerate(METHODS)})
    frame["_dimension"] = pd.to_numeric(frame["dimension"], errors="coerce").fillna(-1)
    return frame.sort_values(["_dataset", "_method", "_dimension"]).drop(columns=["_dataset", "_method", "_dimension"])


def write_classification(out: Path, summary: pd.DataFrame) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    pair = lambda row, metric: "{:.3f} ± {:.3f}".format(row[metric + "_mean"], row[metric + "_std"])
    lines = [
        "# Classification results", "",
        "Generated by `emb3.summarize`. Do not edit by hand.", "",
        "Each cell is the mean over the test folds ± the standard deviation across those folds.",
        "The last column is the half-width of the 95% confidence interval of the mean accuracy (Student t).", "",
    ]
    lines += markdown_table(
        ["Dataset", "Method", "Dimension", "Test folds", "Accuracy", "Macro-F1", "Macro AUC (one-vs-rest)", "Most selected setting", "Accuracy 95% CI"],
        [[row["dataset"], METHOD_NAMES[row["method"]], _dimension(row["dimension"]), row["folds"],
          pair(row, "accuracy"), pair(row, "macro_f1"), pair(row, "macro_auc"),
          "{} ({:.0%} of folds)".format(row["most_selected"], row["most_selected_share"]),
          "± {:.3f}".format(row["accuracy_ci95"])] for _, row in summary.iterrows()],
    )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cost(out: Path, runs: pd.DataFrame, summary: pd.DataFrame) -> None:
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    seconds = lambda value: "{:.2f}".format(value) if value < 100 else "{:,.0f}".format(value)
    lines = [
        "# Compute cost", "",
        "Generated by `emb3.summarize`. Do not edit by hand.", "",
        "Only clean timing runs count: each ran alone, in a fresh process, limited to the thread",
        "budget in the configuration. The measured time excludes library imports, loading the",
        "dataset and converting it to the library's input format. No warm-up run is discarded,",
        "so every run is a cold start. Time is training plus embedding generation for the",
        "method's reference setting; for GIN one run is one cross-validation fold.", "",
        "`Total` is the median over the runs; `95% CI` is the half-width of the confidence",
        "interval of the mean (Student t). Memory is the peak resident set size of the process.", "",
    ]
    lines += markdown_table(
        ["Dataset", "Method", "Dimension", "Setting", "Runs", "Total (s)", "Mean ± std (s)", "95% CI (s)", "Train (s)", "Generate (s)", "Peak memory (GB)"],
        [[row["dataset"], METHOD_NAMES[row["method"]], _dimension(row["dimension"]), row["setting"], row["runs"],
          seconds(row["total_seconds_median"]),
          "{} ± {}".format(seconds(row["total_seconds_mean"]), seconds(row["total_seconds_std"])),
          "± {}".format(seconds(row["total_seconds_ci95"])),
          seconds(row["train_seconds_median"]), seconds(row["generate_seconds_median"]),
          "{:.2f}".format(row["peak_rss_gb_median"])] for _, row in summary.iterrows()],
    )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


CLUSTERING_PROTOCOLS = {
    "test_folds": ("Test graphs of each fold (all methods)",
                   "k-means on the test graphs of every cross-validation fold, so GIN is scored on graphs whose "
                   "labels its model never saw. Each cell is the mean ± standard deviation over the folds."),
    "all_graphs": ("Whole dataset (label-free methods)",
                   "k-means on all graphs at once, once per seed (embedding seed and k-means seed). Each cell is "
                   "the mean ± standard deviation over the seeds."),
}


def clustering_summary(config: Dict[str, Any]) -> pd.DataFrame:
    files = sorted((project_path(config, "results") / "clustering").glob("*/*.csv"))
    if not files:
        return pd.DataFrame()
    runs = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    rows = []
    for (dataset, method, dimension, protocol), group in runs.groupby(["dataset", "method", "dimension", "protocol"], dropna=False, sort=False):
        row: Dict[str, Any] = {"dataset": dataset, "method": method, "dimension": dimension, "protocol": protocol,
                               "setting": group["setting"].iloc[0], "clusterings": len(group), "graphs": group["graphs"].median()}
        for metric in ("ari", "nmi", "nmi_chance", "nmi_size"):
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_std"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def write_clustering(out: Path, summary: pd.DataFrame) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    pair = lambda row, metric: "{:.3f} ± {:.3f}".format(row[metric + "_mean"], row[metric + "_std"])
    lines = [
        "# Clustering results", "",
        "Generated by `emb3.summarize`. Do not edit by hand.", "",
        "k-means with k = number of classes on each method's reference setting; the WL kernel uses",
        "kernel k-means (k-means in its feature space). Vectors are standardised first. ARI is 0 for a",
        "random partition and 1 for the classes exactly. NMI is not corrected for chance: `NMI by chance`",
        "is the NMI of the same clusters with their members shuffled, so only the gap above it counts.",
        "`NMI with size` compares the clusters with graph size instead of class (k groups of equal count",
        "by number of nodes): a high value means k-means sorted the graphs by size.",
    ]
    for protocol, (title, explanation) in CLUSTERING_PROTOCOLS.items():
        part = summary[summary["protocol"] == protocol]
        if not len(part):
            continue
        lines += ["", "## " + title, "", explanation, ""]
        lines += markdown_table(
            ["Dataset", "Method", "Dimension", "Setting", "Clusterings", "Graphs each", "ARI", "NMI", "NMI by chance", "NMI with size"],
            [[row["dataset"], METHOD_NAMES[row["method"]], _dimension(row["dimension"]), row["setting"], row["clusterings"],
              "{:,.0f}".format(row["graphs"]), pair(row, "ari"), pair(row, "nmi"), "{:.3f}".format(row["nmi_chance_mean"]),
              "{:.3f}".format(row["nmi_size_mean"])]
             for _, row in part.iterrows()],
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_clustering_ablation(config: Dict[str, Any], out: Path) -> int:
    """results/clustering/ablation.{csv,md} from the files of emb3.cluster_ablation."""
    files = sorted(out.glob("ablation_*.csv"))
    if not files:
        return 0
    runs = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    order = {name: i for i, name in enumerate(config["datasets"])}
    runs["_dataset"] = runs["dataset"].map(order)
    rows = []
    for (_, dataset, method, variant), group in runs.groupby(["_dataset", "dataset", "method", "variant"], sort=False):
        rows.append({"dataset": dataset, "method": method, "variant": variant, "main": bool(group["main"].iloc[0]),
                     "seeds": len(group), "ari_mean": group["ari"].mean(), "ari_std": group["ari"].std(ddof=1),
                     "nmi_mean": group["nmi"].mean(), "nmi_size_mean": group["nmi_size"].mean(),
                     "silhouette_mean": group["silhouette"].mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "ablation.csv", index=False)
    lines = [
        "# Clustering: preprocessing variants", "",
        "Generated by `emb3.summarize` from `emb3.cluster_ablation`. Do not edit by hand.", "",
        "Whole dataset, k = number of classes, mean over seeds. Every label-free variant is listed; the one",
        "marked *main* is the setting of the main clustering results, fixed before any result was seen.",
        "None is chosen by its agreement with the labels. `NMI with size` compares the clusters with graph",
        "size; the silhouette (no labels) shows how cleanly the clusters separate in the space clustered.",
    ]
    for dataset, part in summary.groupby("dataset", sort=False):
        lines += ["", "## " + dataset, ""]
        lines += markdown_table(
            ["Method", "Variant", "ARI", "NMI", "NMI with size", "Silhouette"],
            [[METHOD_NAMES[row["method"]], row["variant"] + (" (main)" if row["main"] else ""),
              "{:.3f} ± {:.3f}".format(row["ari_mean"], row["ari_std"]), "{:.3f}".format(row["nmi_mean"]),
              "{:.3f}".format(row["nmi_size_mean"]), "{:.3f}".format(row["silhouette_mean"])] for _, row in part.iterrows()],
        )
    (out / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(summary)


def write_stability(config: Dict[str, Any], out: Path) -> int:
    """results/stability/summary.{csv,md} from the files of emb3.stability."""
    from emb3.perturb import variants

    overlap_files = sorted(out.glob("*/*_overlap.csv"))
    accuracy_files = sorted(out.glob("*/*_accuracy.csv"))
    if not overlap_files or not accuracy_files:
        return 0
    overlaps = pd.concat([pd.read_csv(path) for path in overlap_files], ignore_index=True)
    accuracies = pd.concat([pd.read_csv(path) for path in accuracy_files], ignore_index=True)
    names = [(name, "{} {:.0%}".format(mode, fraction)) for name, mode, fraction in variants(config)]
    order = {name: i for i, name in enumerate(config["datasets"])}
    method_order = {name: i for i, name in enumerate(METHOD_NAMES)}
    rows = []
    for (dataset, method), group in overlaps.groupby(["dataset", "method"], sort=False):
        row: Dict[str, Any] = {"dataset": dataset, "method": method, "setting": group["setting"].iloc[0],
                               "neighbours": int(group["neighbours"].iloc[0])}
        lookup = group.set_index("comparison")["overlap_mean"]
        row["overlap_reference"] = lookup.get("reference", np.nan)
        # Two random neighbour lists of k among n - 1 graphs share k^2 / (n - 1) graphs on average.
        k, others = row["neighbours"], int(group["graphs"].iloc[0]) - 1
        row["overlap_random"] = (k * k / others) / (2 * k - k * k / others)
        folds = accuracies[(accuracies["dataset"] == dataset) & (accuracies["method"] == method)]
        base = folds[folds["variant"] == "original"].set_index(["repeat", "fold"])["accuracy"]
        row["accuracy_original"] = base.mean()
        row["accuracy_original_std"] = base.std(ddof=1)
        row["folds"] = len(base)
        for name, _ in names:
            row["overlap_" + name] = lookup.get(name, np.nan)
            changed = folds[folds["variant"] == name].set_index(["repeat", "fold"])["accuracy"]
            difference = (changed - base).dropna()
            row["accuracy_change_" + name] = difference.mean() if len(difference) else np.nan
            row["accuracy_change_std_" + name] = difference.std(ddof=1) if len(difference) > 1 else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary.assign(_d=summary["dataset"].map(order), _m=summary["method"].map(method_order)).sort_values(["_d", "_m"]).drop(columns=["_d", "_m"])
    summary.to_csv(out / "summary.csv", index=False)

    number = lambda value: "-" if pd.isna(value) else "{:.3f}".format(value)
    points = lambda value, spread: "-" if pd.isna(value) else "{:+.1f} ± {:.1f}".format(100 * value, 100 * spread if not pd.isna(spread) else 0.0)
    lines = [
        "# Stability under edge perturbation", "",
        "Generated by `emb3.summarize` from `emb3.stability`. Do not edit by hand.", "",
        "Each method in its reference setting at {} dimensions (the WL kernel has none). Perturbed copies add or".format(config["stability"]["dimension"]),
        "remove the stated share of every graph's edges (`emb3.perturb`).", "",
        "## Neighbourhood overlap", "",
        "Mean Jaccard overlap, over all graphs, of each graph's {} nearest neighbours in two representations.".format(int(summary["neighbours"].iloc[0])),
        "`Reference`: the original data with two different seeds, the change caused by randomness alone (1 for the",
        "deterministic FGSD and WL kernel). The other columns: original against the perturbed copy, same seed.",
        "`Random lists`: the overlap two random neighbour lists would have, for scale.", "",
    ]
    lines += markdown_table(
        ["Dataset", "Method", "Reference (seed vs seed)"] + [label for _, label in names] + ["Random lists"],
        [[row["dataset"], METHOD_NAMES[row["method"]], number(row["overlap_reference"])] + [number(row["overlap_" + name]) for name, _ in names]
         + ["{:.4f}".format(row["overlap_random"])] for _, row in summary.iterrows()],
    )
    lines += [
        "", "## Classification accuracy", "",
        "Test accuracy on the original data and its change on each perturbed copy, in percentage points: mean ±",
        "standard deviation of the per-fold difference over the same folds. Model selection as in the main",
        "classification, but only among the candidates that share the reference representation (they differ in the",
        "SVM's C, and for FGSD in counts against shares), since only that representation is recomputed on the",
        "perturbed copies; GIN keeps its reference setting, and the large dataset uses the reference setting as is.", "",
    ]
    lines += markdown_table(
        ["Dataset", "Method", "Folds", "Original accuracy"] + [label for _, label in names],
        [[row["dataset"], METHOD_NAMES[row["method"]], row["folds"],
          "{:.3f} ± {:.3f}".format(row["accuracy_original"], row["accuracy_original_std"])]
         + [points(row["accuracy_change_" + name], row["accuracy_change_std_" + name]) for name, _ in names]
         for _, row in summary.iterrows()],
    )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(summary)


# Figure styling. Three hues for the three embedding methods, checked for colour-vision
# deficiencies as a set; the baseline is drawn in neutral ink, not as a fourth hue.
SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"graph2vec": ("#2a78d6", "o"), "fgsd": ("#eb6834", "s"), "gin": ("#1baf7a", "^")}


def _seconds_label(value: float, _position: Any = None) -> str:
    if value >= 1:
        return "{:,.0f}".format(value)
    return "{:g}".format(value)


def accuracy_vs_cost_figure(accuracy: pd.DataFrame, cost: pd.DataFrame, datasets: List[str], path: Path) -> Optional[Path]:
    """One panel per dataset: test accuracy against time to train and embed, on a log scale.
    Each method is a line through its embedding dimensions; the WL kernel is a reference line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

    merged = accuracy.merge(cost, on=["dataset", "method", "dimension"], how="inner")
    datasets = [name for name in datasets if (merged["dataset"] == name).any()]
    if not datasets:
        return None

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10})
    figure, axes = plt.subplots(1, len(datasets), figsize=(4.7 * len(datasets), 4.4), squeeze=False, facecolor=SURFACE)
    labelled: List[Any] = []
    for axis, dataset in zip(axes[0], datasets):
        panel = merged[merged["dataset"] == dataset]
        axis.set_facecolor(SURFACE)
        axis.set_xscale("log")
        baseline = panel[panel["method"] == "wl_kernel"]
        if len(baseline):
            level = baseline["accuracy_mean"].iloc[0]
            axis.axhline(level, color=AXIS, linewidth=1, zorder=1)
            axis.plot(baseline["total_seconds_median"], baseline["accuracy_mean"], marker="D", markersize=8, linestyle="none",
                      color=INK_SECONDARY, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
            axis.annotate("WL kernel", xy=(baseline["total_seconds_median"].iloc[0], level), xytext=(8, 6),
                          textcoords="offset points", ha="left", va="bottom", fontsize=8.5, color=INK_SECONDARY)
        for method, (colour, marker) in SERIES.items():
            series = panel[panel["method"] == method].sort_values("dimension")
            if not len(series):
                continue
            x, y = series["total_seconds_median"], series["accuracy_mean"]
            axis.errorbar(x, y, yerr=series["accuracy_std"], fmt="none", ecolor=colour, elinewidth=1, alpha=0.45, zorder=2)
            axis.plot(x, y, color=colour, linewidth=2, solid_capstyle="round", solid_joinstyle="round", marker=marker,
                      markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            labelled.append((axis, series))
        axis.set_title(dataset, loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
        axis.set_xlabel("Time to train and embed (seconds, log scale)", color=INK_SECONDARY)
        axis.xaxis.set_major_formatter(FuncFormatter(_seconds_label))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        axis.grid(True, which="major", color=GRID, linewidth=1)
        axis.set_axisbelow(True)
        axis.minorticks_off()
        axis.margins(x=0.12, y=0.18)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(AXIS)
        axis.tick_params(colors=INK_MUTED, length=0, labelsize=9)
    axes[0][0].set_ylabel("Test accuracy", color=INK_SECONDARY)

    handles = [Line2D([], [], color=colour, linewidth=2, marker=marker, markersize=8, markeredgecolor=SURFACE,
                      markeredgewidth=2, label=METHOD_NAMES[method]) for method, (colour, marker) in SERIES.items()]
    handles.append(Line2D([], [], color=AXIS, linewidth=1, marker="D", markersize=8, markerfacecolor=INK_SECONDARY,
                          markeredgecolor=SURFACE, markeredgewidth=2, label="WL kernel (baseline)"))
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=len(handles), frameon=False,
                  labelcolor=INK_SECONDARY, fontsize=9.5, handlelength=2.2, columnspacing=1.8)
    figure.text(0.01, 0.01, "Each line runs through the embedding dimensions 32, 128 and 512; the ends are labelled. Time is the median of the clean timing runs; "
                "accuracy is the mean over test folds, and the bars show its standard deviation.",
                fontsize=8.5, color=INK_MUTED, ha="left", va="bottom")
    figure.tight_layout(rect=(0, 0.045, 1, 0.93))

    # Dimension labels, placed once the layout is final. Only the smallest and the largest
    # dimension are labelled, off to the side so the error bars stay clear. When a method's
    # cost does not depend on the dimension its points coincide and get one joint label.
    figure.canvas.draw()
    for axis, series in labelled:
        first, last = series.iloc[0], series.iloc[-1]
        ends = axis.transData.transform([(first["total_seconds_median"], first["accuracy_mean"]),
                                         (last["total_seconds_median"], last["accuracy_mean"])])
        note = dict(textcoords="offset points", fontsize=8, color=INK_MUTED)
        if len(series) > 1 and np.hypot(*(ends[1] - ends[0])) < 0.2 * figure.dpi:
            top = series.loc[series["accuracy_mean"].idxmax()]
            axis.annotate("{}–{}".format(int(first["dimension"]), int(last["dimension"])),
                          xy=(top["total_seconds_median"], top["accuracy_mean"]), xytext=(9, 0), ha="left", va="center", **note)
            continue
        axis.annotate(str(int(first["dimension"])), xy=(first["total_seconds_median"], first["accuracy_mean"]),
                      xytext=(-6, 6), ha="right", va="bottom", **note)
        if len(series) > 1:
            axis.annotate(str(int(last["dimension"])), xy=(last["total_seconds_median"], last["accuracy_mean"]),
                          xytext=(6, 6), ha="left", va="bottom", **note)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


def main() -> None:
    config = load_config()
    results = project_path(config, "results")
    folds = classification_folds(config)
    runs = cost_runs(config)
    accuracy = _ordered(classification_summary(folds), config) if len(folds) else pd.DataFrame()
    cost = _ordered(cost_summary(runs), config) if len(runs) else pd.DataFrame()
    if len(accuracy):
        write_classification(results / "classification", accuracy)
    if len(cost):
        write_cost(results / "cost", runs, cost)
    figure = None
    if len(accuracy) and len(cost):
        for frame in (accuracy, cost):
            frame["dimension"] = pd.to_numeric(frame["dimension"], errors="coerce")
        figure = accuracy_vs_cost_figure(accuracy, cost, list(config["datasets"]), results / "figures" / "accuracy_vs_cost.png")
    clustering = clustering_summary(config)
    if len(clustering):
        write_clustering(results / "clustering", _ordered(clustering, config))
    write_clustering_ablation(config, results / "clustering")
    stability_rows = write_stability(config, results / "stability")
    print("classification rows: {}, cost rows: {}, clustering rows: {}, stability rows: {}, figure: {}".format(
        len(accuracy), len(cost), len(clustering), stability_rows, figure or "not enough data yet"))


if __name__ == "__main__":
    main()
