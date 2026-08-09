import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .selection import matches as _matches


def _read_history(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    history = defaultdict(list)
    for row in rows:
        try:
            values = {key: float(row[key]) for key in row}
        except (TypeError, ValueError):
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


def _plot(curves, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.get_cmap("tab10")
    for index, (label, history) in enumerate(curves):
        color = colors(index % 10)
        epochs = history["epoch"]
        for ax, metric in zip(axes, ("loss", "dice")):
            ax.plot(epochs, history[f"train_{metric}"], color=color, ls="--", alpha=0.6)
            ax.plot(epochs, history[f"val_{metric}"], color=color, label=label)
    for ax, metric in zip(axes, ("loss", "dice")):
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} (dashed=train, solid=val)")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize="small")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _render(results_dir: Path, datasets, experiments):
    written = []
    for dataset_dir, dataset_runs in sorted(_collect(results_dir, datasets, experiments).items()):
        curves = []
        for label, (history, fold_dir) in sorted(dataset_runs.items()):
            out_path = fold_dir / "history.png"
            _plot([(label, history)], f"{dataset_dir.name} {label}", out_path)
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
