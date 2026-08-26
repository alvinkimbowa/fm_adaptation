import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


from . import agreement
from .datasets import dataset_dir as resolve_dataset_dir, resolve
from .metrics import read_case_metrics
from .selection import matches


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
    rows = read_case_metrics(path)
    return {
        "dice": np.array([float(row["dice"]) for row in rows]),
        "masd": np.array([float(row["masd"]) for row in rows]),
        "cases": np.array([row["case_id"] for row in rows]),
    }


# How each plans variant is written in the tables, where the directory name reads badly.
NNUNET_VARIANT_NAMES = {"ResEncUNetM": "Res Enc M"}
# Configurations that are their own network rather than a flavour of nnU-Net, so they carry no nnU-Net
# prefix. The xtiny widths (`xtiny8`, `xtiny32`) share one name: no dataset was trained with more than
# one of them, and the Params column already separates them.
NNUNET_MODEL_NAMES = {r"xtiny\d*": "XTinyUNet"}


def nnunet_label(trainer_dir_name):
    """`nnUNetTrainer__nnUNetResEncUNetMPlans__2d` -> `nnU-Net (Res Enc M)`.

    Each plans/configuration pair is a different network -- the Res Enc M and the xtiny plans differ by
    two orders of magnitude in size -- so each earns its own row. Whatever distinguishes the directory
    from the default `nnUNetPlans__2d` becomes the label, and the stock 2d plan stays plain `nnU-Net`.
    """
    _, _, rest = trainer_dir_name.partition("__")
    plans, _, configuration = rest.partition("__")
    variant = plans.removeprefix("nnUNet").removesuffix("Plans")
    variant = NNUNET_VARIANT_NAMES.get(variant, variant)
    configuration = configuration.removeprefix("2d").lstrip("_")
    for pattern, name in NNUNET_MODEL_NAMES.items():
        if re.fullmatch(pattern, configuration):
            return name
    suffix = " ".join(filter(None, (variant, configuration)))
    return f"nnU-Net ({suffix})" if suffix else "nnU-Net"


def _add_nnunet_records(records, results_dir):
    metrics_paths = sorted(
        Path(results_dir).glob("nnunet/Dataset*/*/fold_*/test/Dataset*/metrics.csv")
    )
    if not metrics_paths:
        raise RuntimeError(f"No nnU-Net metrics found under {results_dir}")
    for metrics_path in metrics_paths:
        fold_dir = metrics_path.parents[2]
        trained_on = fold_dir.parents[1].name
        fold = fold_dir.name.removeprefix("fold_")
        tested_on = metrics_path.parent.name
        model = nnunet_label(fold_dir.parent.name)
        records[(model, "", trained_on, fold)][tested_on] = _read_metrics(metrics_path)


MONOUNET_NAMES = {
    "MonoUNetE123V2GatedDA": "MonoUNet-t",
    "MonoUNetE123V2GatedS8DA": "MonoUNet-B",
    "MonoUNetE123V2GatedS32DA": "MonoUNet-L",
}


def _add_monounet_records(records, results_dir, model="MonoUNet"):
    """MonoUNet stores per-case rows as `test/<Dataset>/image_wise_...csv`, dice as a fraction."""
    results_dir = Path(results_dir)
    metrics_paths = sorted(
        results_dir.glob("Dataset*/fold_*/test/Dataset*/image_wise_results_largest_component.csv")
    )
    if not metrics_paths:
        raise RuntimeError(f"No MonoUNet metrics found under {results_dir}")
    for metrics_path in metrics_paths:
        fold_dir = metrics_path.parents[2]
        trained_on = fold_dir.parent.name
        fold = fold_dir.name.removeprefix("fold_")
        tested_on = metrics_path.parent.name
        records[(model, "", trained_on, fold)][tested_on] = _read_metrics(metrics_path)


def _pool_folds(records, folds):
    """Keeps the requested folds and pools their per-case metrics into a single row."""
    pooled = defaultdict(dict)
    label = ",".join(folds)
    for (model, adaptation, trained_on, fold), results in records.items():
        if fold not in folds:
            continue
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


# The annotators, in the order they read best. Paul is deliberately last: no model trains on him and
# every one scores badly there, so he belongs at the edge of the table rather than in among the
# columns being compared. An annotator not named here sorts after these, alphabetically.
INDIVIDUAL_SOURCES = ("katie", "eric", "mohammad", "yvonne", "paul")

# Marks a column as one annotator's rather than one dataset's. Not a dataset name, so the two can
# never collide.
ANNOTATOR = "\0annotator:"


def _annotator(case_id):
    """Who drew a case, from the source its id names, or None where the id names none.

    `Mohammad__10`, `Yvonne_b2__Cond-Lesion-...` and `Katie_contusion__CLE2-rat1` are Mohammad,
    Yvonne and Katie: what stands before the `__`, up to the first qualifier -- a batch, a model of
    injury -- that says which of their images it is rather than whose it is.
    """
    source, separator, _ = case_id.partition("__")
    return source.split("_")[0] if separator else None


def _column_annotator(metrics):
    """The one annotator every case in a column belongs to, or None where they span several.

    This is what makes a single-annotator evaluation set that annotator's column and leaves the
    interrater set, which holds two annotators' work, standing as a column of its own.
    """
    names = [_annotator(case) for case in metrics["cases"]]
    if not names or any(name is None for name in names):
        return None
    unique = set(names)
    return unique.pop() if len(unique) == 1 else None


def _subset(metrics, keep):
    """The cases of a column that `keep` marks, as a column in its own right."""
    index = np.asarray(keep, dtype=bool)
    return {name: values[index] for name, values in metrics.items()}


def _annotator_columns(records, datasets):
    """Rewrites the evaluation sets into one column per annotator.

    An annotator the model trained on is read off the row's own held-out split, which was cut the
    same way the training was; one it never met keeps its own dataset, which is all of that
    annotator's data. The first is in-domain and the second is transfer, so which a cell is comes
    back with it, for the greying and the average to key off.
    """
    keep = set(datasets)
    rewritten, in_domain, columns = {}, set(), set()
    for key, results in records.items():
        trained_on = key[2]
        rewrite = {}
        own = results.get(trained_on)
        if own is not None:
            names = [_annotator(case) for case in own["cases"]]
            for name in sorted({name for name in names if name is not None}):
                column = ANNOTATOR + name
                rewrite[column] = _subset(own, [held == name for held in names])
                in_domain.add((trained_on, column))
                columns.add(column)
        for dataset, metrics in results.items():
            if dataset not in keep:
                continue
            annotator_name = None if dataset == trained_on else _column_annotator(metrics)
            column = ANNOTATOR + annotator_name if annotator_name else dataset
            columns.add(column)
            rewrite.setdefault(column, metrics)
        rewritten[key] = rewrite
    return rewritten, sorted(columns), in_domain


def _dataset_family(dataset):
    """The second token of the dataset name, which is what puts a run in one table or another."""
    return dataset.split("_", maxsplit=2)[1]


PARAMETER_COUNTS = {}


def _load_parameter_counts(path):
    """Counts are gathered by `count_params.py`; without that file the columns simply read '—'."""
    path = Path(path)
    if path.exists():
        PARAMETER_COUNTS.update(json.loads(path.read_text()))


def _format_count(count):
    """A parameter count in its own unit, so a linear probe and a fine-tuned trunk are both readable.

    Fixing every row in millions puts three orders of magnitude on one scale: MonoUNet-t's 1697
    parameters and a probe's few tens of thousands both round to 0.0, which reads as nothing at all.
    """
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if count >= scale:
            return f"{count / scale:.1f}{suffix}"
    return str(count)


def _parameter_counts(model, adaptation, trained_on):
    """Foundation-model counts depend only on the architecture; baselines vary per dataset."""
    entry = PARAMETER_COUNTS.get(f"{model}|{adaptation}|{trained_on}") or PARAMETER_COUNTS.get(
        f"{model}|{adaptation}|"
    )
    if not entry:
        return "—", "—"
    return _format_count(entry["total"]), _format_count(entry["trainable"])


def _dataset_label(dataset):
    """Drops the `Dataset0xx_` prefix for display; an annotator column reads as the annotator."""
    if dataset == OWN_TEST:
        return "Test"
    if dataset.startswith(ANNOTATOR):
        return dataset.removeprefix(ANNOTATOR)
    return re.sub(r"^Dataset\d+_", "", dataset)


def _config_label(model, adaptation):
    report_model, report_adaptation = _report_names(model, adaptation)
    return " + ".join(value for value in (report_model, report_adaptation) if value)


ADAPTATIONS = {
    "linear": ("LP", 0),
    "nonlinear": ("NLP", 1),
    "linear_finetune": ("LP + FT", 2),
    "nonlinear_finetune": ("NLP + FT", 3),
    "upernet": ("Adapter + UperNet", 4),
    "upernet_inj": ("Adapter + UperNet + Inj", 5),
    "upernet_ours": ("Adapter + UperNet ours", 6),
    "upernet_inj_ours": ("Adapter + UperNet + Inj ours", 7),
    "upernet_inj_ft_ours": ("Adapter + UperNet + Inj + FT ours", 8),
    # The ViT-L trunk on the full-length warmup + poly schedule, directly under its constant-rate,
    # early-stopped counterpart, the way each smaller trunk pairs with its own below.
    "upernet_inj_ft_poly_ours": ("Adapter + UperNet + Inj + FT poly ours", 9),
    "upernet_inj_ft_init_ours": ("Adapter + UperNet + Inj + FT init ours", 10),
    "upernet_inj_ft_dropsmi_ours": ("Adapter + UperNet + Inj + FT drop-SMI ours", 11),
    "upernet_inj_ft_dropany_ours": ("Adapter + UperNet + Inj + FT drop-any ours", 12),
    "upernet_inj_ft_balanced_ours": ("Adapter + UperNet + Inj + FT balanced ours", 13),
    "upernet_inj_ft_balanced_dropany_ours": ("Adapter + UperNet + Inj + FT balanced drop-any ours", 14),
    "m2f": ("Adapter + Mask2Former", 15),
    "m2f_inj": ("Adapter + Mask2Former + Inj", 16),
    # The same adaptation on the smaller trunks, sorted last so they close out each model's block of
    # rows, largest trunk first so the rows read down in decreasing size from the ViT-L above. Each
    # trunk keeps its two schedules together -- constant rate with early stopping, then the full-length
    # warmup + poly run -- so that comparison is between adjacent rows rather than across the block.
    "upernet_inj_ft_vitb_ours": ("Adapter + UperNet + Inj + FT ViT-B ours", 17),
    "upernet_inj_ft_vitb_poly_ours": ("Adapter + UperNet + Inj + FT ViT-B poly ours", 18),
    "upernet_inj_ft_vits_ours": ("Adapter + UperNet + Inj + FT ViT-S ours", 19),
    "upernet_inj_ft_vits_poly_ours": ("Adapter + UperNet + Inj + FT ViT-S poly ours", 20),
    "": ("", 21),
}


# Adaptations shown in the main tables; the rest go to the ablation report.
# `upernet_inj`, `upernet_ours` and the two trunk-size runs each need their own base: as bare suffixes
# of a shorter name they would read as sweeps and be sent to the ablation page, away from the rows they
# exist to be compared against -- the extractor-only one for `upernet_inj`, the injector one for
# `upernet_ours`, the ViT-L one for the ViT-S and ViT-B runs and the early-stopped ViT-S one for the
# scheduled `_vits_poly_` run.
MAIN_ADAPTATIONS = {
    "linear", "linear_finetune", "upernet", "upernet_inj", "upernet_ours", "upernet_inj_ours",
    "upernet_inj_ft_ours", "upernet_inj_ft_poly_ours", "upernet_inj_ft_init_ours", "m2f", "m2f_inj",
    "upernet_inj_ft_dropsmi_ours", "upernet_inj_ft_dropany_ours",
    "upernet_inj_ft_balanced_ours", "upernet_inj_ft_balanced_dropany_ours",
    "upernet_inj_ft_vitb_ours", "upernet_inj_ft_vitb_poly_ours",
    "upernet_inj_ft_vits_ours", "upernet_inj_ft_vits_poly_ours", "",
}


def _model_matches(model, patterns):
    """`--models` matched leniently, since the internal keys are not what anyone types.

    Case, hyphens and underscores are ignored, so `nnunet`, `nnU-Net` and `nnu_net` all name the same
    rows and `monounet-b` finds `MonoUNet-B`. Globs still work: `monounet*` takes all three MonoUNets.
    A model carrying a parenthesised variant is matched on its base name too, so `nnU-Net` still takes
    every plans variant while `nnU-Net (xtiny32)` or `*xtiny*` narrows it to one.
    """
    if not patterns:
        return True

    def normalise(name):
        return re.sub(r"[-_]", "", name).lower()

    patterns = [normalise(pattern) for pattern in patterns]
    names = {normalise(model), normalise(model.split(" (")[0])}
    return any(matches(name, patterns) for name in names)


def _split_adaptation(adaptation):
    """Split a run name into its known base and any sweep suffix (e.g. `_wd1.0`)."""
    for base in sorted(ADAPTATIONS, key=len, reverse=True):
        if adaptation == base:
            return base, ""
        if base and adaptation.startswith(f"{base}_"):
            return base, adaptation[len(base) + 1 :]
    return "", adaptation


def _report_names(model, adaptation):
    if not adaptation:
        return model, ""
    models = {"sam3": "SAM3", "dinov3": "DINOv3"}
    base, suffix = _split_adaptation(adaptation)
    label = ADAPTATIONS[base][0]
    if suffix:
        label = f"{label} ({suffix})" if label else suffix
    return models.get(model, model), label


# Rows are grouped by model first: each foundation model's adaptations together, then the baselines.
# Foundation models are keyed by their config name, baselines by the label their loader records.
MODEL_ORDER = {
    "sam3": 0,
    "dinov3": 1,
    "nnU-Net": 2,
    "XTinyUNet": 3,
    "MonoUNet-L": 4,
    "MonoUNet-B": 5,
    "MonoUNet-t": 6,
}


def _model_rank(model):
    """nnU-Net's plans variants are named `nnU-Net (...)`, so they rank where plain nnU-Net does."""
    if model in MODEL_ORDER:
        return MODEL_ORDER[model]
    base = model.split(" (")[0]
    return MODEL_ORDER.get(base, len(MODEL_ORDER))


def _experiment_order(item):
    model, adaptation, trained_on, fold = item[0]
    base, suffix = _split_adaptation(adaptation)
    return (
        trained_on,
        _model_rank(model),
        model,
        ADAPTATIONS[base][1],
        suffix,
        fold,
    )


def _best_values(records, datasets, reducer, in_domain=(), averaged=(), own_test_column=False):
    """Maps each column of a trained-on group to its (best, second best) values.

    Blanked cells are skipped and the average is taken over the same columns the table averages, so
    the highlighting cannot mark a value the table does not show.
    """
    seen = defaultdict(list)
    rows_per_group = Counter(trained_on for _, _, trained_on, _ in records)
    for (_, _, trained_on, _), results in records.items():
        if rows_per_group[trained_on] < 2:
            continue  # Nothing to compare against, so nothing is "best".
        cross = {"dice": [], "masd": []}
        for dataset in datasets:
            metrics = _column_metrics(results, dataset, trained_on, own_test_column)
            if metrics is None or (trained_on, dataset) in in_domain:
                continue
            for metric in ("dice", "masd"):
                value = _reduce(metrics[metric], reducer)
                if np.isnan(value):
                    continue
                seen[(trained_on, dataset, metric)].append(value)
                if dataset in averaged and dataset != trained_on:
                    cross[metric].append(value)
        for metric, values in cross.items():
            if not values:
                continue
            value = _reduce(values, reducer)
            if not np.isnan(value):
                seen[(trained_on, "cross", metric)].append(value)
    ranked = {}
    for key, values in seen.items():
        ordered = sorted(set(values), reverse=key[2] == "dice")
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


# Evaluation sets that earn their own table rather than a column among the transfer results. The
# interrater set is the only place two annotators mark the same images, so it is measured against the
# agreement table beside it. Paul's slides are widefield rather than the confocal tiles everything
# else is built from, so they are external in the sense that matters here -- a different imaging
# modality, not just a different annotator -- and as a column they would dominate an average meant to
# compare the rest.
STANDALONE_TABLES = (("interrater", "Interrater set"), ("paul", "External (widefield)"))


def _standalone_table(dataset):
    """The heading this dataset gets its own table under, or None to keep it as a column."""
    lowered = dataset.lower()
    return next((title for key, title in STANDALONE_TABLES if key in lowered), None)


# The stand-in for "whatever this row held out of its own training set". Not a dataset name, so it
# can never collide with one.
OWN_TEST = "\0own-test"


def _collapsed_columns(datasets, records):
    """The training sets that say nothing except on their own row, and so become one `Test` column.

    A training set that some other row was evaluated on is a real transfer target -- the czi_B model
    tested on czi_120 -- and keeps its column. One that nobody else was sent to holds a single value
    per row, on the diagonal, so a table of them is mostly dashes stating one fact each.
    """
    trained_on = {key[2] for key in records}
    return {
        dataset for dataset in datasets
        if dataset in trained_on
        and not any(
            dataset in results and dataset != fitted_to
            for (_, _, fitted_to, _), results in records.items()
        )
    }


def _column_metrics(results, dataset, trained_on, own_test_column=False):
    """What a row shows in one column, or None where it shows nothing.

    A row's result on its own training set belongs in one place. Where the table carries a `Test`
    column it goes there, and the training set's own column is left to the rows that were sent to it.
    """
    if dataset == OWN_TEST:
        return results.get(trained_on)
    if own_test_column and dataset == trained_on:
        return None
    return results.get(dataset)


def _order_columns(datasets, records):
    """Training sets first, then interrater, then the per-annotator sets, then anything else.

    A row's own training set and the other rows' training sets belong together on the left: reading
    across, you see what each model was fitted to before you see where it was sent. The per-annotator
    columns are what the cross-dataset average is taken over, so they sit together on the right.
    """
    trained_on = {key[2] for key in records}
    def rank(dataset):
        if dataset in trained_on:
            return (0, dataset)
        if "interrater" in dataset:
            return (1, dataset)
        if dataset.startswith(ANNOTATOR):
            name = _dataset_label(dataset).lower()
            rank = INDIVIDUAL_SOURCES.index(name) if name in INDIVIDUAL_SOURCES else len(INDIVIDUAL_SOURCES)
            return (2, rank, dataset)
        return (3, dataset)
    return sorted(datasets, key=lambda d: (rank(d)[0], *rank(d)[1:]))


def _render_table(records, datasets, statistic, in_domain=()):
    fmt = _mean_sd if statistic == "Mean ± SD" else _median_iqr
    reducer = np.mean if statistic == "Mean ± SD" else np.median
    collapsed = _collapsed_columns(datasets, records)
    datasets = _order_columns([d for d in datasets if d not in collapsed], records)
    own_test_column = bool(collapsed)
    if own_test_column:
        datasets = [OWN_TEST] + datasets
    # The average is over the per-annotator columns only -- the combined training sets hold each
    # other's images and the interrater set holds theirs too, so folding those in would weight some
    # images several times over. Averaging a single column just repeats it, so it only appears when
    # there are more.
    averaged = [d for d in datasets if d.startswith(ANNOTATOR)]
    if len(averaged) < 2:
        # A table with no per-annotator columns averages every column but the row's own training set,
        # so that it still reports one rather than nothing at all.
        averaged = [d for d in datasets if d != OWN_TEST]
    show_average = (
        max(
            (
                sum(
                    1 for dataset in averaged
                    if _column_metrics(results, dataset, trained_on, own_test_column) is not None
                    and dataset != trained_on
                    and (trained_on, dataset) not in in_domain
                )
                for (_, _, trained_on, _), results in records.items()
            ),
            default=0,
        )
        > 1
    )
    best = _best_values(records, datasets, reducer, in_domain, averaged, own_test_column)
    parts = [f"<h2>{html.escape(statistic)}</h2><table><thead><tr>"]
    for heading in ("Config", "Params", "Trainable", "Trained on", "Fold"):
        parts.append(f"<th rowspan='2'{_sep(heading == 'Fold')}>{heading}</th>")
    for index, dataset in enumerate(datasets):
        last = index == len(datasets) - 1 and show_average
        parts.append(f"<th colspan='2'{_sep(last)}>{html.escape(_dataset_label(dataset))}</th>")
    if show_average:
        parts.append("<th colspan='2'>Cross-dataset average</th>")
    parts.append("</tr><tr>")
    for index in range(len(datasets) + show_average):
        last = index == len(datasets) - 1 and show_average
        parts.append(f"<th>Dice ↑</th><th{_sep(last)}>MASD (px) ↓</th>")
    parts.append("</tr></thead><tbody>")
    previous_trained_on = None
    for key, results in sorted(records.items(), key=_experiment_order):
        model, probe, trained_on, fold = key
        config = _config_label(model, probe)
        row_class = " class='group-start'" if previous_trained_on not in (None, trained_on) else ""
        total, trainable = _parameter_counts(model, probe, trained_on)
        parts.append(
            f"<tr{row_class}><td>{html.escape(config)}</td>"
            f"<td>{total}</td><td>{trainable}</td>"
            f"<td>{html.escape(_dataset_label(trained_on))}</td>"
            f"<td class='sep'>{html.escape(fold)}</td>"
        )
        previous_trained_on = trained_on
        cross_dice, cross_masd = [], []
        for index, dataset in enumerate(datasets):
            last = index == len(datasets) - 1 and show_average
            metrics = _column_metrics(results, dataset, trained_on, own_test_column)
            if metrics is None:
                parts.append(f"<td>—</td><td{_sep(last)}>—</td>")
                continue
            reference = (trained_on, dataset) in in_domain
            dice_value = _reduce(metrics["dice"], reducer)
            masd_value = _reduce(metrics["masd"], reducer)
            parts.append(
                _metric_cell(
                    fmt(metrics["dice"], 100),
                    dice_value,
                    best.get((trained_on, dataset, "dice")),
                    reference=reference,
                )
            )
            parts.append(
                _metric_cell(
                    fmt(metrics["masd"]),
                    masd_value,
                    best.get((trained_on, dataset, "masd")),
                    separator=last,
                    reference=reference,
                )
            )
            if dataset in averaged and not reference and dataset != trained_on:
                cross_dice.append(dice_value)
                cross_masd.append(masd_value)
        if not show_average:
            pass
        elif cross_dice:
            cross_dice_value = _reduce(cross_dice, reducer)
            cross_masd_value = _reduce(cross_masd, reducer)
            parts.append(
                _metric_cell(
                    fmt(np.asarray(cross_dice), 100),
                    cross_dice_value,
                    best.get((trained_on, "cross", "dice")),
                )
            )
            parts.append(
                _metric_cell(
                    fmt(np.asarray(cross_masd)),
                    cross_masd_value,
                    best.get((trained_on, "cross", "masd")),
                )
            )
        else:
            parts.append("<td>—</td><td>—</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _write_summary_csv(records, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "adaptation",
                "trained_on",
                "fold",
                "tested_on",
                "n",
                "dice_mean",
                "dice_sd",
                "dice_median",
                "dice_q1",
                "dice_q3",
                "masd_mean",
                "masd_sd",
                "masd_median",
                "masd_q1",
                "masd_q3",
            ]
        )
        for key, results in sorted(records.items(), key=_experiment_order):
            model, adaptation, trained_on, fold = key
            report_model, report_adaptation = _report_names(model, adaptation)
            for tested_on, metrics in sorted(results.items()):
                dice, masd = metrics["dice"], metrics["masd"]
                writer.writerow(
                    [
                        report_model,
                        report_adaptation,
                        trained_on,
                        fold,
                        tested_on,
                        len(dice),
                        np.mean(dice),
                        np.std(dice, ddof=1),
                        np.median(dice),
                        *np.percentile(dice, [25, 75]),
                        np.mean(masd),
                        np.inf if np.isinf(masd).any() else np.std(masd, ddof=1),
                        np.median(masd),
                        *np.percentile(masd, [25, 75]),
                    ]
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="models")
    parser.add_argument("--nnunet-results-dir", nargs="*", default=[])
    parser.add_argument(
        "--nnunet-raw-data-dir",
        default=None,
        help="Where the nnU-Net baselines' training datasets live. They carry no config of their "
        "own, so without this their training set has no images to compare an evaluation set "
        "against and every cell of theirs reads as unseen.",
    )
    parser.add_argument("--monounet-results-dir", nargs="*", default=[])
    parser.add_argument(
        "--folds",
        default="0",
        help="Comma-separated folds to compile, pooled into one row (e.g. '0,1'); '' keeps each fold separate",
    )
    parser.add_argument("--output", default="models/cross_dataset_report.html")
    parser.add_argument("--parameter-counts", default="models/parameter_counts.json")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Training datasets to tabulate; empty keeps every one, as in the plotting scripts",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=[],
        help="Run names to tabulate, matched exactly, as a glob or as a `_suffix` tag; empty keeps all",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Models to compare (dinov3, sam3, nnU-Net, MonoUNet-t/B/L), ignoring case and hyphens; "
        "empty keeps every one",
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
    if args.nnunet_raw_data_dir is not None:
        for _, _, trained_on, _ in records:
            raw_dirs.setdefault(trained_on, Path(args.nnunet_raw_data_dir).expanduser())
    for results_dir in args.monounet_results_dir:
        name = Path(results_dir).name
        _add_monounet_records(records, results_dir, MONOUNET_NAMES.get(name, name))
    records = {
        key: results
        for key, results in records.items()
        # `key` is (model, adaptation, trained_on, fold). The baselines carry no adaptation name, so
        # `--experiments` narrows the foundation-model rows and leaves nnU-Net and MonoUNet standing as
        # the comparison; drop those with `--models`.
        if _model_matches(key[0], args.models)
        and matches(key[2], args.datasets)
        and (not key[1] or matches(key[1], args.experiments))
    }
    if args.folds:
        records = _pool_folds(records, [fold.strip() for fold in args.folds.split(",")])
    if not records:
        selection = ", ".join(
            filter(None, (" ".join(args.models), " ".join(args.datasets), " ".join(args.experiments)))
        )
        raise RuntimeError(
            f"No cross-dataset metrics found under {args.results_dir}"
            + (f" for {selection}" if selection else "")
        )
    style = """
    body{background:#111;color:#bbb;font-family:system-ui;margin:16px}table{border-collapse:collapse;width:auto;max-width:100%}
    th,td{padding:7px 10px;text-align:center}th{background:#292929}td{background:#191919;border-bottom:1px solid #222}
    tr.group-start td{border-top:3px solid #777}strong{color:#eee;font-weight:700}
    th.sep,td.sep{border-right:2px solid #777}
    .undef{color:#777;font-size:11px;font-weight:400;margin-top:2px}
    /* Scored on images the model was trained on: kept for reference, greyed so it cannot be misread
       as a transfer result, and excluded from both the ranking and the row average. */
    td.reference{color:#555;font-style:italic}
    u{color:#ddd;text-decoration:underline;text-underline-offset:3px}
    /* Config and "Trained on" read as labels, so they stay left; everything else, the parameter
       counts included, is centred like the metrics. */
    tbody td:first-child,tbody td:nth-child(4),
    thead tr:first-child th:first-child,thead tr:first-child th:nth-child(4){text-align:left}
    section{margin-bottom:56px}h1{color:#ddd;font-size:22px;margin:0 0 18px}h2{font-size:16px;font-weight:400;margin-top:28px}
    """
    # One page per (table kind, statistic); `suffix` becomes part of the file name.
    statistics = {"mean_sd": "Mean ± SD", "median_iqr": "Median (Q1–Q3)"}
    bodies = {(kind, suffix): "" for kind in ("main", "ablation") for suffix in statistics}
    families = sorted({_dataset_family(trained_on) for _, _, trained_on, _ in records})
    for family in families:
        family_records = {
            key: results
            for key, results in records.items()
            if _dataset_family(key[2]) == family
        }
        family_datasets = sorted(
            {
                tested_on
                for results in family_records.values()
                for tested_on in results
                if _dataset_family(tested_on) == family
            }
        )
        # From here the columns are annotators, not datasets. `family_datasets` keeps the dataset
        # names, which is what the agreement section below reads its labels from.
        family_records, family_columns, in_domain = _annotator_columns(family_records, family_datasets)
        swept_bases = {
            _split_adaptation(key[1])[0]
            for key in family_records
            if _split_adaptation(key[1])[1]
        }
        main_records = {
            key: results
            for key, results in family_records.items()
            if _split_adaptation(key[1]) in {(base, "") for base in MAIN_ADAPTATIONS}
        }
        # Everything else — nonlinear probes and sweeps — plus the runs they vary from.
        ablation_records = {
            key: results
            for key, results in family_records.items()
            if key not in main_records or _split_adaptation(key[1])[0] in swept_bases
        }

        # A dataset that ships an interrater split holds the same image annotated twice, which is
        # the ceiling every model in the table above is measured against. Measured by
        # compute_metrics, like every other number here, and read back from what it wrote.
        measured = {}
        for dataset in family_datasets:
            dataset_dir = resolve_dataset_dir(raw_dirs.get(dataset, Path()), dataset)
            for split in agreement.splits(dataset_dir):
                rows, unpaired = agreement.read(
                    agreement.path_for(agreement_dir, dataset_dir.name, split)
                )
                if rows or unpaired:
                    measured[dataset] = (rows, unpaired)

        standalone = defaultdict(list)
        transfer_datasets = []
        for dataset in family_columns:
            title = _standalone_table(dataset)
            (standalone[title] if title else transfer_datasets).append(dataset)
        # Keep the order STANDALONE_TABLES declares rather than whatever the datasets sorted into.
        standalone = [(title, standalone[title]) for _, title in STANDALONE_TABLES if standalone[title]]

        for suffix, statistic in statistics.items():
            bodies[("main", suffix)] += (
                f"<section><h1>{html.escape(family)}</h1>"
                + _render_table(main_records, transfer_datasets, statistic, in_domain)
                + "".join(
                    f"<h1 style='margin-top:40px'>{html.escape(title)}</h1>"
                    + _render_table(main_records, columns, statistic, in_domain)
                    for title, columns in standalone
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
            if ablation_records:
                bodies[("ablation", suffix)] += (
                    f"<section><h1>{html.escape(family)} — Ablation</h1>"
                    + _render_table(ablation_records, transfer_datasets, statistic, in_domain)
                    + "</section>"
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(records, output.with_suffix(".csv"))
    for (kind, suffix), page_body in bodies.items():
        if not page_body:
            continue
        name = f"{output.stem}{'_ablation' if kind == 'ablation' else ''}_{suffix}{output.suffix}"
        path = output.with_name(name)
        path.write_text(f"<!doctype html><meta charset='utf-8'><style>{style}</style>{page_body}")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
