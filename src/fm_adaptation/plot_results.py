"""Per-image results by image-metadata group, and how a training set's images spread over the groups.

Two figures per training set, both cut by one metadata field (`location`, say):

- a box plot of each run's per-image metric on its own held-out split, one box per run inside each
  group, with every image drawn over it as a jittered point;
- a bar chart of how many images each group holds in the fold's train and validation parts and in
  the dataset's own `imagesTs`.

Runs are chosen the way every other figure here chooses them, by the parts of their directory.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .datasets import dataset_dir, family, fold_cases, metadata_values, split_cases
from .metrics import read_case_metrics
from .naming import describe_run
from .config import describe_run_dir
from .selection import select_runs

# Categorical hues in a fixed order, so a run keeps its colour however many are drawn.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
ALL = "all"


def group_cases(cases, values, order):
    """{group: [case ids]} for the groups `order` names, in its order, then `ALL`.

    An empty `order` keeps every value the field takes, sorted. A case counts in each of its groups
    and once in `ALL`, which holds every case in a kept group plus the cases the field says nothing
    about.
    """
    order = list(order) or sorted({v for case in cases for v in values.get(case, ())})
    groups = {group: [] for group in order}
    groups[ALL] = []
    for case in cases:
        given = values.get(case, ())
        kept = [group for group in given if group in groups and group != ALL]
        for group in kept:
            groups[group].append(case)
        if kept or not given:
            groups[ALL].append(case)
    return groups


def _folder(dataset):
    return re.sub(r"[^a-z0-9]+", "_", family(dataset).lower()).strip("_")


def _label(run_dir):
    return describe_run(run_dir.parent.name).removesuffix(" + ours")


def plot_boxes(runs, groups, metric, title, output, seed=0):
    """`runs` is [(label, {case: value})]; one box per run inside each group."""
    rng = np.random.default_rng(seed)
    names = list(groups)
    width = 0.8 / len(runs)
    figure, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(names) * max(1, len(runs)) ** 0.5 + 2), 4.6))
    for index, (label, values) in enumerate(runs):
        color = PALETTE[index % len(PALETTE)]
        offset = (index - (len(runs) - 1) / 2) * width
        for position, group in enumerate(names):
            data = np.array([values[case] for case in groups[group] if case in values], dtype=float)
            data = data[np.isfinite(data)]
            if not data.size:
                continue
            x = position + offset
            ax.boxplot(
                data, positions=[x], widths=width * 0.8, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor=color + "33", edgecolor=color, linewidth=1.2),
                medianprops=dict(color=color, linewidth=2),
                whiskerprops=dict(color=color, linewidth=1), capprops=dict(color=color, linewidth=1),
            )
            jitter = rng.uniform(-width * 0.3, width * 0.3, data.size)
            ax.scatter(x + jitter, data, s=9, color=color, alpha=0.6, linewidths=0, zorder=3)
        ax.plot([], [], color=color, linewidth=6, label=label)
    counts = [len(groups[group]) for group in names]
    ax.set_xticks(range(len(names)), [f"{group}\nn={n}" for group, n in zip(names, counts)])
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=min(len(runs), 2))
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_bars(parts, title, output):
    """`parts` is [(label, {group: count})] with the same groups in each; one bar per part per group."""
    names = [group for group in parts[0][1] if group != ALL]
    width = 0.8 / len(parts)
    figure, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(names) + 2), 4.0))
    for index, (label, counts) in enumerate(parts):
        offset = (index - (len(parts) - 1) / 2) * width
        heights = [counts[group] for group in names]
        bars = ax.bar(np.arange(len(names)) + offset, heights, width * 0.92,
                      color=PALETTE[index % len(PALETTE)], label=f"{label} (n={counts[ALL]})")
        ax.bar_label(bars, fontsize=8, padding=2, color="#444444")
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("Images")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", nargs="*", default=["models"])
    parser.add_argument("--raw-data-dir", default=None,
                        help="where the datasets live, for runs that ship no config.yaml of their own")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--train-datasets", nargs="*", default=[])
    parser.add_argument("--configs", nargs="*", default=[])
    parser.add_argument("--folds", nargs="*", default=["0"])
    parser.add_argument("--split", default="test", choices=("test", "validation"),
                        help="which of a run's own held-out parts the box plot reads")
    parser.add_argument("--metric", default="dice")
    parser.add_argument("--split-by", default="location", help="image-metadata field to group by")
    parser.add_argument("--groups", nargs="*", default=[],
                        help="values of --split-by to show, in this order; empty keeps every value")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--format", default="png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dirs = select_runs(args.results_dir, args.models, args.train_datasets, args.configs,
                           args.folds)
    per_dataset = {}
    for run_dir in run_dirs:
        cfg = describe_run_dir(run_dir, args.raw_data_dir)
        if cfg is None:
            print(f"skipped {run_dir} (no config.yaml, and no --raw-data-dir to stand in for it)")
            continue
        per_dataset.setdefault(cfg.train_dataset, (cfg.raw_data_dir, []))[1].append(run_dir)
    if not per_dataset:
        print("nothing selected")
        return

    for trained_on, (raw_data_dir, dataset_runs) in per_dataset.items():
        directory = dataset_dir(raw_data_dir, trained_on)
        values = metadata_values(directory, args.split_by)
        folder = Path(args.output_dir) / _folder(trained_on)

        runs, cases = [], set()
        for run_dir in dataset_runs:
            path = run_dir / args.split / trained_on / "metrics.csv"
            if not path.is_file():
                print(f"skipped {run_dir.parent.name}: no {args.split} metrics for {trained_on}")
                continue
            rows = read_case_metrics(path)
            runs.append((_label(run_dir), {row["case_id"]: row[args.metric] for row in rows}))
            cases |= set(runs[-1][1])
        if runs:
            output = folder / f"{args.split}__{args.metric}_by_{args.split_by}__{trained_on}.{args.format}"
            plot_boxes(runs, group_cases(sorted(cases), values, args.groups), args.metric,
                       f"{trained_on} — {args.split}, {args.metric} per image", output, args.seed)
            print(f"wrote {output}")

        parts = []
        for fold in dict.fromkeys(run_dir.name.removeprefix("fold_") for run_dir in dataset_runs):
            split = fold_cases(directory, fold)
            if split is not None:
                parts += [(f"Train (fold {fold})", split[0]), (f"Validation (fold {fold})", split[1])]
        test = sorted(split_cases(directory, "Ts"))
        if test:
            parts.append(("Test", test))
        if parts:
            counts = [(label, {group: len(members) for group, members
                               in group_cases(part, values, args.groups).items()})
                      for label, part in parts]
            output = folder / f"training_by_{args.split_by}__{trained_on}.{args.format}"
            plot_bars(counts, f"{trained_on} — images per {args.split_by}", output)
            print(f"wrote {output}")


if __name__ == "__main__":
    main()
