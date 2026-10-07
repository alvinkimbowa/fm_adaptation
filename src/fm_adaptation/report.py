import argparse
import csv
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml


from . import agreement
from .datasets import (
    dataset_dir as resolve_dataset_dir,
    dataset_root,
    family as _dataset_family,
    metadata_values,
    resolve,
    split_cases,
)
from .naming import MODEL_NAMES, describe_run
from .metrics import read_case_metrics
from .selection import list_order, matches


def _read_run(run_dir: Path):
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    raw_data_dir = Path(cfg["data"]["raw_data_dir"])
    return (
        cfg["model"]["name"],
        cfg["model"].get("run_name", cfg["model"]["probe"]),
        resolve(raw_data_dir, cfg["data"]["train_dataset"]),
        raw_data_dir,
    )


def _read_metrics(path: Path):
    """Every metric column the file carries. A run scored before a metric existed simply has no
    column for it, and the tables that ask for it read '\u2014'."""
    rows = read_case_metrics(path)
    metrics = {
        key: np.array([float(row[key]) for row in rows])
        for key in METRICS
        if rows and key in rows[0]
    }
    return {**metrics, "cases": np.array([row["case_id"] for row in rows])}


@dataclass(frozen=True)
class Metric:
    """A metric column: where it comes from, how it reads, and which end of it is good."""

    label: str
    scale: float = 1.0
    higher_is_better: bool = False


# Every metric the tables can show. What each task is judged on is `FAMILY_METRICS` below.
METRICS = {
    "dice": Metric("Dice \u2191", 100, higher_is_better=True),
    "cldice": Metric("clDice \u2191", 100, higher_is_better=True),
    "masd": Metric("MASD (px) \u2193"),
    "hd95": Metric("HD95 (px) \u2193"),
}
# Tracing is judged on whether the same paths were followed, not on where a two-pixel-wide trace
# landed, so the neurite tables carry clDice and no surface distance. Everything else is scored on
# overlap and boundary.
FAMILY_METRICS = {"neurites": ("dice", "cldice")}
DEFAULT_METRICS = ("dice", "masd")


def _family_metrics(family):
    return FAMILY_METRICS.get(family, DEFAULT_METRICS)


# How each plans variant is written in the tables, where the directory name reads badly.
NNUNET_VARIANT_NAMES = {"ResEncUNetM": "Res Enc M"}
# Configurations that are their own network rather than a flavour of nnU-Net, so they carry no nnU-Net
# prefix. The xtiny widths (`xtiny8`, `xtiny32`) share one name: no dataset was trained with more than
# one of them, and the Params column already separates them.
NNUNET_MODEL_NAMES = {r"xtiny\d*": "XTinyUNet"}
# How a MonoUNet architecture directory reads. The directory spells out the variant it was built with
# -- `MonoUNetE123V2GatedDA` -- which is more than a table wants to carry, so the family alone is the
# label. A directory this does not match keeps its own name.
MONOUNET_MODEL_NAMES = {r"MonoUNet\w*": "MonoUNet"}
# How a trainer's tag is written, where the directory name reads badly.
NNUNET_TRAINER_NAMES = {"100epochs": "100 epochs", "SkeletonRecall": "Skeleton Recall"}


def nnunet_label(trainer_dir_name):
    """`nnUNetTrainer_100epochs__nnUNetResEncUNetMPlans__2d` -> `nnU-Net Res Enc M (100 epochs)`.

    The plans is the network: the stock configuration and the residual-encoder preset are trained by
    the same code but are different architectures, so the plans names the model and each is selected
    on its own. What is left -- the trainer's tag, the configuration -- says how that network was
    trained and stays a parenthesised suffix, and the stock `nnUNetTrainer__nnUNetPlans__2d` keeps
    the bare name `nnU-Net`.
    """
    trainer, _, rest = trainer_dir_name.partition("__")
    plans, _, configuration = rest.partition("__")
    variant = plans.removeprefix("nnUNet").removesuffix("Plans")
    variant = NNUNET_VARIANT_NAMES.get(variant, variant)
    configuration = configuration.removeprefix("2d").lstrip("_")
    trainer = " ".join(
        NNUNET_TRAINER_NAMES.get(part, part)
        for part in trainer.removeprefix("nnUNetTrainer").split("_")
        if part
    )
    name = f"nnU-Net {variant}" if variant else "nnU-Net"
    for pattern, model_name in NNUNET_MODEL_NAMES.items():
        if re.fullmatch(pattern, configuration):
            name, configuration = model_name, ""
            break
    suffix = ", ".join(filter(None, (configuration, trainer)))
    return f"{name} ({suffix})" if suffix else name


def monounet_label(architecture_dir_name):
    """`MonoUNetE123V2GatedDA` -> `MonoUNet`, by the vocabulary in `MONOUNET_MODEL_NAMES`."""
    for pattern, model_name in MONOUNET_MODEL_NAMES.items():
        if re.fullmatch(pattern, architecture_dir_name):
            return model_name
    return architecture_dir_name


def _add_nnunet_records(records, results_dir):
    results_dir = Path(results_dir)
    metrics_paths = sorted(results_dir.glob("nnunet/Dataset*/*/fold_*/test/Dataset*/metrics.csv"))
    # A training dataset shipping no `imagesTs` is held out by fold instead, and nnU-Net writes that
    # fold's predictions flat into `validation/` rather than under a directory named after the set.
    own_paths = sorted(results_dir.glob("nnunet/Dataset*/*/fold_*/validation/metrics.csv"))
    if not metrics_paths and not own_paths:
        raise RuntimeError(f"No nnU-Net metrics found under {results_dir}")
    for metrics_path in metrics_paths:
        fold_dir = metrics_path.parents[2]
        trained_on = fold_dir.parents[1].name
        fold = fold_dir.name.removeprefix("fold_")
        tested_on = metrics_path.parent.name
        model = nnunet_label(fold_dir.parent.name)
        records[(model, "", trained_on, fold)][tested_on] = _read_metrics(metrics_path)
    for metrics_path in own_paths:
        fold_dir = metrics_path.parents[1]
        trained_on = fold_dir.parents[1].name
        # The training dataset is reported once, in a single column: on its own imagesTs when the run
        # produced one, otherwise on the fold it held out.
        if (fold_dir / "test" / trained_on / "metrics.csv").exists():
            continue
        fold = fold_dir.name.removeprefix("fold_")
        model = nnunet_label(fold_dir.parent.name)
        records[(model, "", trained_on, fold)][trained_on] = _read_metrics(metrics_path)


def _add_monounet_records(records, results_dir):
    """Rows from a MonoUNet architecture directory, whose layout is the whole of what it says.

    `<architecture>/<trained on>/fold_N/test/<tested on>/metrics.csv`: the architecture takes the
    place the nnU-Net trees give a trainer, and, as for those, the configuration slot stays empty --
    a baseline is selected by `--models` alone. Two directories that read as the same model would
    otherwise silently overwrite each other, so the collision is raised instead.
    """
    results_dir = Path(results_dir)
    metrics_paths = sorted(results_dir.glob("Dataset*/fold_*/test/Dataset*/metrics.csv"))
    if not metrics_paths:
        raise RuntimeError(f"No MonoUNet metrics found under {results_dir}")
    model = monounet_label(results_dir.name)
    for metrics_path in metrics_paths:
        fold_dir = metrics_path.parents[2]
        trained_on = fold_dir.parent.name
        fold = fold_dir.name.removeprefix("fold_")
        key = (model, "", trained_on, fold)
        tested_on = metrics_path.parent.name
        if tested_on in records[key]:
            raise RuntimeError(f"{results_dir} repeats a row already read as {model}: {key}")
        records[key][tested_on] = _read_metrics(metrics_path)


def _pool_folds(records, folds):
    """Keeps the requested folds and pools their per-case metrics into a single row.

    The row is labelled with the folds it turned out to hold, so a run that has only fold 0 says so
    rather than reading as an average over folds it never trained.
    """
    pooled = defaultdict(dict)
    held = defaultdict(set)
    for (model, adaptation, trained_on, fold) in records:
        if fold in folds:
            held[(model, adaptation, trained_on)].add(fold)
    for (model, adaptation, trained_on, fold), results in records.items():
        if fold not in folds:
            continue
        label = ",".join(sorted(held[(model, adaptation, trained_on)]))
        target = pooled[(model, adaptation, trained_on, label)]
        for tested_on, metrics in results.items():
            if tested_on not in target:
                target[tested_on] = metrics
                continue
            target[tested_on] = {
                name: np.concatenate([target[tested_on][name], values])
                for name, values in metrics.items()
            }
    return pooled


def _finite(values, scale=1.0):
    """Splits off the nan/inf cases (empty prediction, missing surface) from the usable ones."""
    values = np.asarray(values, dtype=float) * scale
    finite = values[np.isfinite(values)]
    return finite, len(values) - len(finite)


def _reduce(values, reducer):
    finite, _ = _finite(values)
    return reducer(finite) if len(finite) else float("nan")


def _annotate(text, undefined):
    if not undefined:
        return text
    return f"{text}<div class='undef'>({undefined} undef.)</div>"


def _mean_sd(values, scale=1.0):
    finite, undefined = _finite(values, scale)
    if not len(finite):
        return _annotate("—", undefined)
    ddof = 1 if len(finite) > 1 else 0
    return _annotate(f"{finite.mean():.2f} ± {finite.std(ddof=ddof):.2f}", undefined)


def _median_iqr(values, scale=1.0):
    finite, undefined = _finite(values, scale)
    if not len(finite):
        return _annotate("—", undefined)
    q1, median, q3 = np.percentile(finite, [25, 50, 75])
    return _annotate(f"{median:.2f} ({q1:.2f}–{q3:.2f})", undefined)


def _render_interrater(rows, unpaired, statistic):
    """Human agreement, grouped by which two annotators drew the pair."""
    fmt = _mean_sd if statistic == "Mean ± SD" else _median_iqr
    parts = [f"<h2>{html.escape(statistic)}</h2><table><thead><tr>"
             "<th>Annotators</th><th class='sep'>Image</th>"
             "<th>Dice ↑</th><th>MASD (px) ↓</th></tr></thead><tbody>"]
    groups = defaultdict(list)
    for annotators, name, dice, masd in rows:
        groups[" | ".join(annotators)].append((name, dice, masd))
    for group, entries in sorted(groups.items()):
        for index, (name, dice, masd) in enumerate(sorted(entries)):
            start = " class='group-start'" if index == 0 else ""
            parts.append(
                f"<tr{start}><td>{html.escape(group) if index == 0 else ''}</td>"
                f"<td class='sep'>{html.escape(_shorten_name(name))}</td>"
                # One image is one measurement, so it is printed as the value it is; the spread
                # belongs to the group row underneath.
                f"<td>{dice * 100:.2f}</td><td>{masd:.2f}</td></tr>"
            )
        dices = [dice for _, dice, _ in entries]
        masds = [masd for _, _, masd in entries]
        parts.append(
            f"<tr><td></td><td class='sep'><strong>{len(entries)} images</strong></td>"
            f"<td><strong>{fmt(dices, 100)}</strong></td><td><strong>{fmt(masds)}</strong></td></tr>"
        )
    parts.append("</tbody></table>")
    if unpaired:
        # An annotation with no counterpart cannot be an agreement measurement; say so rather than
        # dropping it, since it usually means the split is missing a file.
        names = ", ".join(html.escape(case) for case in unpaired)
        parts.append(f"<p class='undef'>unpaired, not measured: {names}</p>")
    return "".join(parts)


def _shorten_name(name, limit=46):
    return name if len(name) <= limit else f"{name[: limit - 1]}…"



PARAMETER_COUNTS = {}


def _load_parameter_counts(path):
    """Counts are gathered by `count_params.py`; without that file the columns simply read '—'."""
    path = Path(path)
    if path.exists():
        PARAMETER_COUNTS.update(json.loads(path.read_text()))


def _format_count(count):
    """A parameter count in its own unit, so a linear probe and a fine-tuned trunk are both readable.

    Fixing every row in millions puts three orders of magnitude on one scale, where a probe's few
    thousand parameters round to 0.0 and read as nothing at all.
    """
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if count >= scale:
            return f"{count / scale:.1f}{suffix}"
    return str(count)


def _parameter_entry(model, adaptation, trained_on):
    """Foundation-model counts depend only on the architecture; baselines vary per dataset."""
    return PARAMETER_COUNTS.get(f"{model}|{adaptation}|{trained_on}") or PARAMETER_COUNTS.get(
        f"{model}|{adaptation}|"
    )


def _parameter_counts(model, adaptation, trained_on):
    entry = _parameter_entry(model, adaptation, trained_on)
    if not entry:
        return "—", "—"
    return _format_count(entry["total"]), _format_count(entry["trainable"])


def _dataset_label(dataset):
    """A column heading: the row's own held-out split, the average, else the dataset's short name."""
    if dataset == OWN_TEST:
        return "Test"
    if dataset == CROSS:
        return "Cross-dataset average"
    return re.sub(r"^Dataset\d+_", "", dataset)


def _config_label(model, adaptation):
    """A row heading, read off the run's own name -- see `naming.describe_run`."""
    return " + ".join(
        value for value in (MODEL_NAMES.get(model, model), describe_run(adaptation)) if value
    )


def _model_matches(model, patterns):
    """`--models` matched leniently, since the internal keys are not what anyone types.

    Case, hyphens, underscores and spaces are ignored, so `nnunet`, `nnU-Net` and `nnu_net` all name
    the same rows, and `nnUNetResEncM` names `nnU-Net Res Enc M` without needing to be quoted. A
    model carrying a parenthesised suffix is matched on the name before it too, so `nnU-Net` takes
    every way the stock plans was trained while `nnUNet*` widens to the other plans as well.
    """
    if not patterns:
        return True

    def normalise(name):
        return re.sub(r"[-_ ]", "", name).lower()

    patterns = [normalise(pattern) for pattern in patterns]
    names = {normalise(model), normalise(model.split(" (")[0])}
    return any(matches(name, patterns) for name in names)


# Columns rows can be ordered by, mapped to the field each reads out of `parameter_counts.json`.
# A name here is what `--sort-by` accepts.
ROW_SORT_COLUMNS = {"params": "total", "trainable": "trainable"}


# Model families, in the order their rows read best: foundation models keyed by their config name,
# baselines by the label their loader records. A model not named here sorts last.
def _experiment_order(models, train_datasets, configs, group_by_train_dataset=True, sort_by="",
                      descending=False):
    """Sort key putting rows where the selection lists put them, or where `sort_by` does.

    Grouped reports put the training set first; ungrouped reports follow the run-directory order of
    model, configuration, then training set. A part whose selection list is empty was not narrowed
    and has no order of its own; `list_order` falls back to the value itself. Naming a sort column
    puts it ahead of all of that within a group, leaving the lists to break ties.
    """
    field = ROW_SORT_COLUMNS.get(sort_by)

    def model_order(model):
        # `--models` is matched leniently, so ordering has to be too: a row labelled
        # `nnU-Net (Res Enc M)` is the one `nnUNet` named and belongs where that entry sits.
        for index, pattern in enumerate(models):
            if _model_matches(model, [pattern]):
                return (index, "")
        return (len(models), model)

    def size_order(model, adaptation, trained_on):
        """Where the sort column puts a row; empty, and so no order at all, when none was named.

        A run whose size was never counted has no place on the scale and sorts after every run that
        does, whichever way round the sort runs.
        """
        if field is None:
            return ()
        entry = _parameter_entry(model, adaptation, trained_on)
        if not entry:
            return (1, 0)
        return (0, -entry[field] if descending else entry[field])

    def key(item):
        model, adaptation, trained_on, fold = item[0]
        size = size_order(model, adaptation, trained_on)
        run_order = (
            model_order(model),
            list_order(adaptation, configs),
            fold,
        )
        dataset_order = list_order(trained_on, train_datasets)
        if group_by_train_dataset:
            return (dataset_order, size, *run_order)
        return (size, *run_order, dataset_order)

    return key


# Datasets built from the same underlying images under different case ids, which no comparison of
# ids can discover. Every member is in-domain against every other, whichever one a run trained on.
SHARED_IMAGES = (
    frozenset({"Dataset203_neurites_yvonne_smi_2px_scaleaug", "Dataset301_neurite_yvonne_b2_smi",
               "Dataset302_neurite_yvonne_b2_smi_1px",
               "Dataset304_neurite_yvonne_b2_smi_1px_scaleaug"}),
)


def _shares_images(dataset):
    """The declared group `dataset` belongs to, or nothing when it stands on its own."""
    return next((group for group in SHARED_IMAGES if dataset in group), frozenset())


def _in_domain(records, datasets, raw_dirs):
    """(training set, evaluation set) pairs scored on images the training set also holds.

    A cell for such a pair is shown for reference but kept out of the ranking and the average. Case
    ids answer this wherever two sets name the same image the same way; where they do not,
    `SHARED_IMAGES` says so instead.
    """
    def cases(dataset):
        directory = resolve_dataset_dir(raw_dirs.get(dataset, Path()), dataset)
        return split_cases(directory, "Tr") | split_cases(directory, "Ts")

    pairs = set()
    for trained_on in {key[2] for key in records}:
        training_cases = cases(trained_on)
        declared = _shares_images(trained_on)
        for dataset in datasets:
            if dataset in declared or (training_cases and training_cases & cases(dataset)):
                pairs.add((trained_on, dataset))
    return pairs


# Evaluation sets that earn their own table rather than a column among the transfer results. The
# interrater set is the only place two annotators mark the same images, so it is measured against the
# agreement table beside it. Paul's slides are widefield rather than the confocal tiles everything
# else is built from, so they are external in the sense that matters here -- a different imaging
# modality, not just a different annotator -- and as a column they would dominate an average meant to
# compare the rest.
# The stand-in for "whatever this row held out of its own training set". Not a dataset name, so it
# can never collide with one, and the same is true of the two below.
OWN_TEST = "\0own-test"
# The average across evaluation sets, which is a section of columns like a dataset is.
CROSS = "\0cross"
# Where the cases a split field says nothing about are collected.
UNGROUPED = "\0ungrouped"


class Column(NamedTuple):
    """One column of results: an evaluation set, and which slice of it the cell measures.

    `group` is empty for the whole set, and otherwise one value of the field a split was asked for.
    """

    dataset: str
    group: str = ""


@dataclass
class Split:
    """How a results column is cut up: one column per value of an image-metadata field.

    An empty `key` leaves each evaluation set whole, which is every table not asked for a split.
    `order` narrows the values and puts them in the order it names them; empty keeps every value the
    field takes, sorted. A case the field says nothing about is collected in a column of its own
    rather than dropped, so the columns still account for every case the metrics hold.
    """

    key: str = ""
    order: tuple[str, ...] = ()
    raw_dirs: dict = dataclass_field(default_factory=dict)
    roots: tuple = ()
    _values: dict = dataclass_field(default_factory=dict)

    def _case_values(self, dataset):
        """One field for every case of a dataset, found wherever that dataset's images live.

        A baseline carries no config of its own, so the root recorded against it is whatever its
        loader was pointed at -- which for a dataset in another project does not hold it. Every root
        any run named is searched, which is what gives the baselines their breakdown too.
        """
        if dataset not in self._values:
            recorded = self.raw_dirs.get(dataset)
            roots = [root for root in (recorded, *self.roots) if root is not None]
            directory = (
                resolve_dataset_dir(dataset_root(roots, dataset), dataset) if roots else Path()
            )
            self._values[dataset] = metadata_values(directory, self.key)
        return self._values[dataset]

    def label(self, group):
        return f"no {self.key}" if group == UNGROUPED else group

    def groups(self, dataset, cases):
        """Which values the field takes over `cases`, before `order` narrows them."""
        if not self.key:
            return {""}
        values = self._case_values(dataset)
        found = set()
        for case in cases:
            answers = values.get(case) or ()
            found |= set(answers) if answers else {UNGROUPED}
        return found

    def arrange(self, groups):
        """The groups a table shows, in order, then all of them together. The ungrouped cases come
        after the named ones, and are never hidden."""
        # A dataset recording nothing for the field is not split: it keeps the single pooled column
        # it has without one.
        if not self.key or not groups - {UNGROUPED, ""}:
            return ("",)
        named = (
            [value for value in self.order if value in groups]
            if self.order
            else sorted(group for group in groups if group != UNGROUPED)
        )
        return tuple(named + ([UNGROUPED] if UNGROUPED in groups else []) + [""])

    def select(self, dataset, values, group):
        """`values` cut down to the cases in one group, or None where it holds none of them.

        The whole-set group `""` keeps the cases of every group the table shows, so values `order`
        leaves out are left out of it too.
        """
        if not self.key or (not group and not self.order):
            return values
        answers = self._case_values(dataset)
        if not group:
            keep = [
                not answers.get(case) or bool(set(answers[case]) & set(self.order))
                for case in values["cases"]
            ]
        else:
            keep = [
                not answers.get(case) if group == UNGROUPED else group in (answers.get(case) or ())
                for case in values["cases"]
            ]
        keep = np.array(keep)
        if not keep.any():
            return None
        return {name: column[keep] for name, column in values.items()}


def _column_source(column, trained_on):
    """Which evaluation set a column reads, `Test` being the row's own held-out split."""
    return trained_on if column.dataset == OWN_TEST else column.dataset


def _column_metrics(results, column, trained_on, split=Split()):
    """What a row shows in one column, or None where it shows nothing.

    `Test` is the row's own held-out split. Every other column is the evaluation set it names, which
    for a set the run trained on is that set's `imagesTs` -- so a run trained on one dataset repeats
    its `Test` value in that dataset's column. A split column reads the same metrics cut down to the
    cases its group holds.
    """
    source = _column_source(column, trained_on)
    values = results.get(source)
    if values is None:
        return None
    return split.select(source, values, column.group)


def _columns(records, datasets, split):
    """One column per evaluation set, or one per slice of it when a split was asked for."""
    columns = []
    for dataset in datasets:
        found = set()
        for (_, _, trained_on, _), results in records.items():
            source = _column_source(Column(dataset), trained_on)
            values = results.get(source)
            if values is not None:
                found |= split.groups(source, values["cases"])
        columns += [Column(dataset, group) for group in split.arrange(found)]
    return columns


def _sections(columns, average_groups, split_blocks=False):
    """The columns grouped into the blocks one heading spans, and where a rule is drawn between them.

    One block per evaluation set, then the average. The rule closing a block is drawn before the
    average always, and between the evaluation sets only where each of them holds several columns --
    an unsplit table is one column per set and reads better without them.
    """
    sections = [
        (dataset, [column for column in columns if column.dataset == dataset])
        for dataset in dict.fromkeys(column.dataset for column in columns)
    ]
    if average_groups:
        sections.append((CROSS, [Column(CROSS, group) for group in average_groups]))
    return [
        (dataset, section, index < len(sections) - 1
         and (split_blocks or sections[index + 1][0] == CROSS))
        for index, (dataset, section) in enumerate(sections)
    ]


def _plan(sections):
    """Each column in reading order, with whether its last metric closes a block of columns."""
    return [
        (column, closes and position == len(section) - 1)
        for _, section, closes in sections
        for position, column in enumerate(section)
    ]


def _best_values(
    records,
    columns,
    reducer,
    in_domain=(),
    averaged=(),
    metrics=DEFAULT_METRICS,
    group_by_train_dataset=True,
    split=Split(),
):
    """Maps each column to its (best, second best) values within each comparison group.

    Blanked cells are skipped and the average is taken over the same columns the table averages, so
    the highlighting cannot mark a value the table does not show. A grouped report compares only
    rows trained on the same dataset; an ungrouped report compares every row.
    """
    seen = defaultdict(list)
    if len(records) < 2:
        return {}
    for (_, _, trained_on, _), results in records.items():
        group = trained_on if group_by_train_dataset else None
        cross = defaultdict(lambda: defaultdict(list))
        for column in columns:
            values = _column_metrics(results, column, trained_on, split)
            if values is None or (trained_on, column.dataset) in in_domain:
                continue
            for metric in metrics:
                if metric not in values:
                    continue
                value = _reduce(values[metric], reducer)
                if np.isnan(value):
                    continue
                seen[(group, column, metric)].append(value)
                if column in averaged:
                    cross[column.group][metric].append(value)
        for slice_, by_metric in cross.items():
            for metric, values in by_metric.items():
                value = _reduce(values, reducer)
                if not np.isnan(value):
                    seen[(group, Column(CROSS, slice_), metric)].append(value)
    ranked = {}
    for key, values in seen.items():
        ordered = sorted(set(values), reverse=METRICS[key[2]].higher_is_better)
        ranked[key] = (ordered[0], ordered[1] if len(ordered) > 1 else None)
    return ranked


def _metric_cell(text, value, ranking, separator=False, reference=False):
    """`reference` greys the cell: shown for context, but outside the ranking and the average."""
    if reference:
        return f"<td class='reference{' sep' if separator else ''}'>{text}</td>"
    # `ranking` is None for groups with a single row, where there is nothing to win against.
    best, second = ranking if ranking else (None, None)
    tag = ""
    if best is not None and np.isclose(value, best, equal_nan=False):
        tag = "strong"
    elif second is not None and np.isclose(value, second, equal_nan=False):
        tag = "u"
    if tag:
        head, marker, tail = text.partition("<div")  # Mark the value, not the undef. note.
        text = f"<{tag}>{head}</{tag}>{marker}{tail}"
    return f"<td{_sep(separator)}>{text}</td>"


def _sep(separator):
    """Marks the last column of a table section (setup | per-dataset results | average)."""
    return " class='sep'" if separator else ""


def _render_table(
    records,
    datasets,
    statistic,
    order,
    in_domain=(),
    metrics=DEFAULT_METRICS,
    group_by_train_dataset=True,
    split=Split(),
):
    fmt = _mean_sd if statistic == "Mean \u00b1 SD" else _median_iqr
    reducer = np.mean if statistic == "Mean \u00b1 SD" else np.median
    # Every row reports its own held-out split under `Test`, whatever else it is shown against.
    columns = _columns(records, [OWN_TEST] + list(datasets), split)
    # `Test` is a different set of images on every row, so it is never averaged.
    averaged = [column for column in columns if column.dataset != OWN_TEST]
    show_average = (
        max(
            (
                len({
                    column.dataset
                    for column in averaged
                    if _column_metrics(results, column, trained_on, split) is not None
                    and (trained_on, column.dataset) not in in_domain
                })
                for (_, _, trained_on, _), results in records.items()
            ),
            default=0,
        )
        > 1
    )
    # One average per slice, so a split table averages wrist against wrist rather than across the
    # slices it was drawn to hold apart.
    average_groups = split.arrange({column.group for column in averaged}) if show_average else ()
    best = _best_values(
        records,
        columns,
        reducer,
        in_domain,
        averaged,
        metrics,
        group_by_train_dataset,
        split,
    )
    # Whether this table's own columns were cut up: a family whose datasets record nothing for the
    # split field is tabulated exactly as it is without one, down to the headings.
    splitting = any(column.group for column in columns)
    sections = _sections(columns, average_groups, splitting)
    plan = _plan(sections)
    # A split table names the evaluation set, then the slice, then the metric; without one the slice
    # row would carry nothing and is left out.
    heading_rows = 3 if splitting else 2
    parts = [f"<h2>{html.escape(statistic)}</h2><table><thead><tr>"]
    for heading in ("Config", "Params", "Trainable", "Trained on", "Fold"):
        parts.append(f"<th rowspan='{heading_rows}'{_sep(heading == 'Fold')}>{heading}</th>")
    for dataset, section, closes in sections:
        parts.append(
            f"<th colspan='{len(metrics) * len(section)}'{_sep(closes)}>"
            f"{html.escape(_dataset_label(dataset))}</th>"
        )
    if splitting:
        parts.append("</tr><tr>")
        for column, separator in plan:
            parts.append(
                f"<th colspan='{len(metrics)}'{_sep(separator)}>"
                f"{html.escape(split.label(column.group) or 'all')}</th>"
            )
    parts.append("</tr><tr>")
    for _, separator in plan:
        for position, metric in enumerate(metrics):
            last = separator and position == len(metrics) - 1
            parts.append(f"<th{_sep(last)}>{METRICS[metric].label}</th>")
    parts.append("</tr></thead><tbody>")
    previous_trained_on = None
    for key, results in sorted(records.items(), key=order):
        model, probe, trained_on, fold = key
        config = _config_label(model, probe)
        total, trainable = _parameter_counts(model, probe, trained_on)
        row_class = (
            " class='group-start'"
            if group_by_train_dataset and previous_trained_on not in (None, trained_on)
            else ""
        )
        parts.append(
            f"<tr{row_class}><td>{html.escape(config)}</td>"
            f"<td>{total}</td><td>{trainable}</td>"
            f"<td>{html.escape(_dataset_label(trained_on))}</td>"
            f"<td class='sep'>{html.escape(fold)}</td>"
        )
        previous_trained_on = trained_on
        ranking_group = trained_on if group_by_train_dataset else None
        cross = defaultdict(lambda: defaultdict(list))
        for column, separator in plan:
            if column.dataset == CROSS:
                for position, metric in enumerate(metrics):
                    last = separator and position == len(metrics) - 1
                    pooled = cross[column.group][metric]
                    if not pooled:
                        parts.append(f"<td{_sep(last)}>\u2014</td>")
                        continue
                    parts.append(
                        _metric_cell(
                            fmt(np.asarray(pooled), METRICS[metric].scale),
                            _reduce(pooled, reducer),
                            best.get((ranking_group, column, metric)),
                            separator=last,
                        )
                    )
                continue
            values = _column_metrics(results, column, trained_on, split)
            reference = values is not None and (trained_on, column.dataset) in in_domain
            for position, metric in enumerate(metrics):
                last = separator and position == len(metrics) - 1
                if values is None or metric not in values:
                    parts.append(f"<td{_sep(last)}>\u2014</td>")
                    continue
                value = _reduce(values[metric], reducer)
                parts.append(
                    _metric_cell(
                        fmt(values[metric], METRICS[metric].scale),
                        value,
                        best.get((ranking_group, column, metric)),
                        separator=last,
                        reference=reference,
                    )
                )
                if column in averaged and not reference:
                    cross[column.group][metric].append(value)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _write_summary_csv(records, path, order, split=Split()):
    """One row per row of the tables and evaluation set, or per slice of it under a split.

    `group` is what the split field said about the cases the row summarises, and is empty where the
    whole evaluation set is one row.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "adaptation",
                "trained_on",
                "fold",
                "tested_on",
                "group",
                "n",
                *(f"{metric}_{statistic}" for metric in METRICS
                  for statistic in ("mean", "sd", "median", "q1", "q3")),
            ]
        )
        for key, results in sorted(records.items(), key=order):
            model, adaptation, trained_on, fold = key
            report_model = MODEL_NAMES.get(model, model)
            report_adaptation = describe_run(adaptation)
            for tested_on, all_cases in sorted(results.items()):
                for group in split.arrange(split.groups(tested_on, all_cases["cases"])):
                    values = split.select(tested_on, all_cases, group)
                    if values is None:
                        continue
                    summary = []
                    for metric in METRICS:
                        # A run scored before the metric existed has no column, and leaves blanks.
                        if metric not in values:
                            summary += [""] * 5
                            continue
                        column = values[metric]
                        summary += [
                            np.mean(column),
                            np.inf if np.isinf(column).any() else np.std(column, ddof=1),
                            np.median(column),
                            *np.percentile(column, [25, 75]),
                        ]
                    writer.writerow(
                        [
                            report_model,
                            report_adaptation,
                            trained_on,
                            fold,
                            tested_on,
                            split.label(group),
                            len(values["dice"]),
                            *summary,
                        ]
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="models")
    parser.add_argument("--nnunet-results-dir", nargs="*", default=[])
    parser.add_argument(
        "--monounet-results-dir",
        nargs="*",
        default=[],
        help="MonoUNet architecture directories, each read as one model",
    )
    parser.add_argument(
        "--nnunet-raw-data-dir",
        default=None,
        help="Where the nnU-Net baselines' training datasets live. They carry no config of their "
        "own, so without this their training set has no images to compare an evaluation set "
        "against and every cell of theirs reads as unseen.",
    )
    parser.add_argument(
        "--folds",
        nargs="*",
        default=["0"],
        help="Folds to compile; several are pooled into one row. Empty keeps each fold separate",
    )
    parser.add_argument("--output", default="models/cross_dataset_report.html")
    parser.add_argument("--parameter-counts", default="models/parameter_counts.json")
    # One list per part of a run directory, `<model>/<train dataset>/<config>/fold_<n>`. Each is
    # matched exactly, as a glob, as a `_suffix` tag or, for a dataset, by its number; empty keeps
    # every value that part can take.
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Models to tabulate, ignoring case and hyphens so `nnunet` finds `nnU-Net`",
    )
    parser.add_argument(
        "--train-datasets", nargs="*", default=[], help="What a run was trained on",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[],
        help="Configurations to tabulate. A baseline carries none and is selected by --models alone",
    )
    parser.add_argument(
        "--test-datasets",
        nargs="*",
        default=[],
        help="Evaluation sets to show as columns, in the order given; empty keeps every one. The "
        "`Test` column is always present and is never averaged.",
    )
    parser.add_argument(
        "--split-by",
        default="",
        help="Cut every results column into one column per value of this image-metadata field -- "
        "`location` for where along the nerve a scan was taken, `anatomy` for what was traced in "
        "it. Empty keeps one column per evaluation set.",
    )
    parser.add_argument(
        "--split-groups",
        nargs="*",
        default=[],
        help="Which values of --split-by to show, in the order given; empty keeps every value the "
        "field takes, sorted",
    )
    parser.add_argument(
        "--group-by-train-dataset",
        type=int,
        choices=(0, 1),
        default=0,
        help="Whether to group rows and rank best values within each training dataset (default: 0)",
    )
    parser.add_argument(
        "--sort-by",
        default="",
        choices=("", *ROW_SORT_COLUMNS),
        help="Order rows within each group by this column; empty follows the selection lists",
    )
    parser.add_argument(
        "--sort-descending",
        type=int,
        choices=(0, 1),
        default=0,
        help="Largest first when --sort-by names a column (default: 0)",
    )
    args = parser.parse_args()
    _load_parameter_counts(args.parameter_counts)
    records = defaultdict(dict)
    # Where compute_metrics wrote the agreement between annotators.
    agreement_dir = Path(args.results_dir) / "agreement"
    # Where each training dataset's images live, so the annotator-agreement section can read the
    # labels themselves; the tables need only the metrics CSVs.
    raw_dirs = {}
    for metrics_path in sorted(Path(args.results_dir).glob("*/*/*/fold_*/*/*/metrics.csv")):
        run_dir = metrics_path.parents[2]
        model, probe, trained_on, raw_data_dir = _read_run(run_dir)
        raw_dirs.setdefault(trained_on, raw_data_dir)
        fold = run_dir.name.removeprefix("fold_")
        tested_on = metrics_path.parent.name
        # Evaluation sets need an entry too: the annotator-agreement section reads their labels, and
        # an interrater set that is not also a training set would otherwise have nowhere to look.
        raw_dirs.setdefault(tested_on, raw_data_dir)
        kind = metrics_path.parents[1].name
        # The training dataset is reported once, in a single column: on its own imagesTs when the run
        # produced one, otherwise on the fold's validation split.
        if tested_on == trained_on and kind == "validation":
            if (run_dir / "test" / tested_on / "metrics.csv").exists():
                continue
        records[(model, probe, trained_on, fold)][tested_on] = _read_metrics(metrics_path)
    for results_dir in args.nnunet_results_dir:
        _add_nnunet_records(records, results_dir)
    for results_dir in args.monounet_results_dir:
        _add_monounet_records(records, results_dir)
    if args.nnunet_raw_data_dir is not None:
        for _, _, trained_on, _ in records:
            raw_dirs.setdefault(trained_on, Path(args.nnunet_raw_data_dir).expanduser())
    records = {
        key: results
        for key, results in records.items()
        if _model_matches(key[0], args.models)
        and matches(key[2], args.train_datasets)
        # A baseline carries no configuration, so --configs cannot name it and --models decides.
        and (not key[1] or matches(key[1], args.configs))
    }
    if args.folds:
        records = _pool_folds(records, [fold.strip() for fold in args.folds])
    if not records:
        selection = " ".join(args.models + args.train_datasets + args.configs)
        raise RuntimeError(
            f"No cross-dataset metrics found under {args.results_dir}"
            + (f" for {selection}" if selection else "")
        )
    style = """
    body{background:#111;color:#bbb;font-family:system-ui;margin:16px}table{border-collapse:collapse;width:auto;max-width:100%}
    th,td{padding:7px 10px;text-align:center}th{background:#292929}td{background:#191919;border-bottom:1px solid #222}
    strong{color:#eee;font-weight:700}
    th.sep,td.sep{border-right:2px solid #777}
    .undef{color:#777;font-size:11px;font-weight:400;margin-top:2px}
    /* Scored on images the model was trained on: kept for reference, greyed so it cannot be misread
       as a transfer result, and excluded from both the ranking and the row average. */
    td.reference{color:#555;font-style:italic}
    tr.group-start td{border-top:3px solid #777}
    u{color:#ddd;text-decoration:underline;text-underline-offset:3px}
    /* Config and "Trained on" read as labels, so they stay left; everything else, the parameter
       counts included, is centred like the metrics. */
    tbody td:first-child,tbody td:nth-child(4),
    thead tr:first-child th:first-child,thead tr:first-child th:nth-child(4){text-align:left}
    section{margin-bottom:56px}h1{color:#ddd;font-size:22px;margin:0 0 18px}h2{font-size:16px;font-weight:400;margin-top:28px}
    """
    split = Split(
        args.split_by, tuple(args.split_groups), raw_dirs, tuple(dict.fromkeys(raw_dirs.values()))
    )
    order = _experiment_order(
        args.models, args.train_datasets, args.configs, bool(args.group_by_train_dataset),
        args.sort_by, bool(args.sort_descending),
    )
    # One page per statistic; `suffix` becomes part of the file name.
    statistics = {"mean_sd": "Mean ± SD", "median_iqr": "Median (Q1–Q3)"}
    bodies = {suffix: "" for suffix in statistics}
    families = sorted({_dataset_family(trained_on) for _, _, trained_on, _ in records})
    for family in families:
        family_records = {
            key: results
            for key, results in records.items()
            if _dataset_family(key[2]) == family
        }
        evaluated = {
            tested_on
            for results in family_records.values()
            for tested_on in results
            if _dataset_family(tested_on) == family
        }
        # Column order follows `--datasets`; an empty selection takes everything, sorted.
        family_datasets = (
            [d for d in args.test_datasets if d in evaluated]
            if args.test_datasets
            else sorted(evaluated)
        )
        in_domain = _in_domain(family_records, family_datasets, raw_dirs)
        # A dataset that ships an interrater split holds the same image annotated twice, which is
        # the ceiling every model in the table above is measured against. Measured by
        # compute_metrics, like every other number here, and read back from what it wrote.
        measured = {}
        for dataset in sorted(evaluated):
            dataset_dir = resolve_dataset_dir(raw_dirs.get(dataset, Path()), dataset)
            for split in agreement.splits(dataset_dir):
                rows, unpaired = agreement.read(
                    agreement.path_for(agreement_dir, dataset_dir.name, split)
                )
                if rows or unpaired:
                    measured[dataset] = (rows, unpaired)

        for suffix, statistic in statistics.items():
            bodies[suffix] += (
                f"<section><h1>{html.escape(family)}</h1>"
                + _render_table(
                    family_records,
                    family_datasets,
                    statistic,
                    order,
                    in_domain,
                    _family_metrics(family),
                    bool(args.group_by_train_dataset),
                    split,
                )
                + "".join(
                    f"<h1 style='margin-top:40px'>Annotator agreement — "
                    f"{html.escape(_dataset_label(dataset))}</h1>"
                    "<p class='undef'>The same image annotated twice. Paired by image content: a "
                    "pair is one slide, though the two annotators did not always work from the same "
                    "export of it.</p>"
                    + _render_interrater(rows, unpaired, statistic)
                    for dataset, (rows, unpaired) in sorted(measured.items())
                )
                + "</section>"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(records, output.with_suffix(".csv"), order, split)
    for suffix, page_body in bodies.items():
        if not page_body:
            continue
        path = output.with_name(f"{output.stem}_{suffix}{output.suffix}")
        path.write_text(f"<!doctype html><meta charset='utf-8'><style>{style}</style>{page_body}")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
