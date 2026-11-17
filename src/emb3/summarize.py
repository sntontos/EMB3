"""Builds the result tables and the accuracy-versus-cost figure from whatever has been run
so far (main environment):

    scripts/run.sh main -m emb3.summarize

Classification: results/classification/summary.{csv,md}, from the per-fold files written
by emb3.classify. Cost: results/cost/{runs,summary}.csv and summary.md, from the job
records of clean timing runs only. Figure: results/figures/accuracy_vs_cost.png.
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
    print("classification rows: {}, cost rows: {}, figure: {}".format(len(accuracy), len(cost), figure or "not enough data yet"))


if __name__ == "__main__":
    main()
