import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .selection import matches as _matches


def _read_history(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    history = defaultdict(list)
    for row in rows:
        try:
            # Runs that validate every few epochs leave blanks, as do the columns mmseg never logs;
            # keep them as NaN so every column stays the same length as the epoch axis. `_series`
            # drops them again at plot time.
            values = {key: float(value) if value else float("nan") for key, value in row.items()}
        except (TypeError, ValueError):
            break
        if not values.get("epoch") == values.get("epoch"):  # epoch itself must be a number
            break
        for key, value in values.items():
            history[key].append(value)
    return history


def _collect(results_dir: Path, datasets, experiments):
    runs = defaultdict(dict)
    for path in sorted(results_dir.glob("*/Dataset*/*/fold_*/history.csv")):
        fold_dir = path.parent
        dataset_dir = fold_dir.parents[1]
        if not _matches(dataset_dir.name, datasets):
            continue
        if not _matches(fold_dir.parent.name, experiments):
            continue
        history = _read_history(path)
        if history.get("epoch"):
            label = f"{fold_dir.parent.name}/{fold_dir.name}"
            runs[dataset_dir][label] = (history, fold_dir)
    return runs


def _moving_average(values, window):
    """Trailing mean over the last `window` points; the first few average whatever precedes them."""
    cumulative = np.cumsum(np.insert(np.asarray(values, dtype=float), 0, 0.0))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def _smoothed(epochs, values):
    """The logged points with gaps dropped, and their mean over the last epoch.

    Runs that check several times per epoch average that many points; runs that check once per
    epoch or less often get a window of one, which leaves the values as logged. Returns None when
    nothing was logged."""
    points = [(x, y) for x, y in zip(epochs, values) if y == y]
    if not points:
        return None
    xs, ys = (np.array(column) for column in zip(*points))
    window = max(1, round(1 / np.median(np.diff(xs)))) if len(xs) > 1 else 1
    return xs, ys, _moving_average(ys, window), window


def _series(ax, epochs, values, label=None, **style):
    """Plot a metric's one-epoch moving average, with every logged value faint behind it. Gaps left
    by runs that validate every few epochs are dropped, so the points still join up."""
    smoothed = _smoothed(epochs, values)
    if smoothed is None:
        return
    xs, ys, average, window = smoothed
    if window > 1:
        ax.plot(xs, ys, lw=0.8, **{**style, "alpha": 0.2})
    marker = "o" if len(xs) < len(epochs) else None
    ax.plot(xs, average, marker=marker, markersize=3, lw=1.8, label=label, **style)


def _mark_best(axes, epochs, val_dice, color):
    """Star the peak of the val Dice moving average and line up its epoch on both panels."""
    smoothed = _smoothed(epochs, val_dice)
    if smoothed is None:
        return
    xs, _, average, _ = smoothed
    best = int(np.argmax(average))
    for ax in axes:
        ax.axvline(xs[best], color=color, ls=":", lw=1, alpha=0.7)
    axes[1].plot(xs[best], average[best], "*", color=color, markersize=12, markeredgecolor="white",
                 label=f"best val {average[best]:.4f} @ epoch {xs[best]:g}")


def _plot(curves, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # One run is coloured by split; several runs share the axes, so each run gets a colour and the
    # split is told apart by line style.
    single = len(curves) == 1
    colors = plt.get_cmap("tab10")
    for index, (label, history) in enumerate(curves):
        epochs = history["epoch"]
        for ax, metric in zip(axes, ("loss", "dice")):
            if single:
                _series(ax, epochs, history[f"train_{metric}"], label="train", color="tab:orange")
                _series(ax, epochs, history[f"val_{metric}"], label="val", color="tab:blue")
            else:
                color = colors(index % 10)
                _series(ax, epochs, history[f"train_{metric}"], color=color, ls="--")
                _series(ax, epochs, history[f"val_{metric}"], label=label, color=color)
        _mark_best(axes, epochs, history["val_dice"], "black" if single else colors(index % 10))
    splits = "" if single else " (dashed=train, solid=val)"
    for ax, metric in zip(axes, ("loss", "dice")):
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}{splits}\nbold: mean over the last epoch, faint: every check", fontsize="medium")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize="small")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_run(run_dir: Path, history=None):
    """Draw one run's own curves next to its history.csv, reading the file when no history is given.

    Returns the path written, or None when the run has not finished an epoch yet."""
    if history is None:
        path = run_dir / "history.csv"
        history = _read_history(path) if path.exists() else {}
    if not history.get("epoch"):
        return None
    label = f"{run_dir.parent.name}/{run_dir.name}"
    out_path = run_dir / "history.png"
    _plot([(label, history)], f"{run_dir.parents[1].name} {label}", out_path)
    return out_path


def _render(results_dir: Path, datasets, experiments):
    written = []
    for dataset_dir, dataset_runs in sorted(_collect(results_dir, datasets, experiments).items()):
        curves = []
        for label, (history, fold_dir) in sorted(dataset_runs.items()):
            out_path = render_run(fold_dir, history)
            written.append((out_path, len(history["epoch"])))
            curves.append((label, history))
        out_path = dataset_dir / "history.png"
        _plot(curves, dataset_dir.name, out_path)
        written.append((out_path, sum(len(h["epoch"]) for h, _ in dataset_runs.values())))
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="models")
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--experiments", nargs="*", default=[])
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    while True:
        for out_path, epochs in _render(results_dir, args.datasets, args.experiments):
            print(f"wrote {out_path} ({epochs} epochs)")
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
