import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from .selection import matches


def _read_run(run_dir: Path):
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    return (
        cfg["model"]["name"],
        cfg["model"].get("run_name", cfg["model"]["probe"]),
        cfg["data"]["train_dataset"],
    )


def _read_metrics(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "dice": np.array([float(row["dice"]) for row in rows]),
        "masd": np.array([float(row["masd"]) for row in rows]),
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


def _dataset_family(dataset):
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
    """Drops the `Dataset0xx_` prefix for display."""
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
    "m2f": ("Adapter + Mask2Former", 11),
    "m2f_inj": ("Adapter + Mask2Former + Inj", 12),
    # The same adaptation on the smaller trunks, sorted last so they close out each model's block of
    # rows, largest trunk first so the rows read down in decreasing size from the ViT-L above. Each
    # trunk keeps its two schedules together -- constant rate with early stopping, then the full-length
    # warmup + poly run -- so that comparison is between adjacent rows rather than across the block.
    "upernet_inj_ft_vitb_ours": ("Adapter + UperNet + Inj + FT ViT-B ours", 13),
    "upernet_inj_ft_vitb_poly_ours": ("Adapter + UperNet + Inj + FT ViT-B poly ours", 14),
    "upernet_inj_ft_vits_ours": ("Adapter + UperNet + Inj + FT ViT-S ours", 15),
    "upernet_inj_ft_vits_poly_ours": ("Adapter + UperNet + Inj + FT ViT-S poly ours", 16),
    "": ("", 17),
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


def _best_values(records, datasets, reducer):
    """Maps each column of a trained-on group to its (best, second best) values."""
    seen = defaultdict(list)
    rows_per_group = Counter(trained_on for _, _, trained_on, _ in records)
    for (_, _, trained_on, _), results in records.items():
        if rows_per_group[trained_on] < 2:
            continue  # Nothing to compare against, so nothing is "best".
        cross = {"dice": [], "masd": []}
        for dataset in datasets:
            metrics = results.get(dataset)
            if metrics is None:
                continue
            for metric in ("dice", "masd"):
                value = _reduce(metrics[metric], reducer)
                if np.isnan(value):
                    continue
                seen[(trained_on, dataset, metric)].append(value)
                if dataset != trained_on:
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


def _metric_cell(text, value, ranking, separator=False):
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


def _render_table(records, datasets, statistic):
    fmt = _mean_sd if statistic == "Mean ± SD" else _median_iqr
    reducer = np.mean if statistic == "Mean ± SD" else np.median
    best = _best_values(records, datasets, reducer)
    # Averaging one external dataset just repeats its column, so only show it when there are more.
    show_average = (
        max(
            (
                sum(1 for dataset in datasets if dataset != trained_on and dataset in results)
                for (_, _, trained_on, _), results in records.items()
            ),
            default=0,
        )
        > 1
    )
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
            metrics = results.get(dataset)
            if metrics is None:
                parts.append(f"<td>—</td><td{_sep(last)}>—</td>")
                continue
            dice_value = _reduce(metrics["dice"], reducer)
            masd_value = _reduce(metrics["masd"], reducer)
            parts.append(
                _metric_cell(
                    fmt(metrics["dice"], 100),
                    dice_value,
                    best.get((trained_on, dataset, "dice")),
                )
            )
            parts.append(
                _metric_cell(
                    fmt(metrics["masd"]),
                    masd_value,
                    best.get((trained_on, dataset, "masd")),
                    separator=last,
                )
            )
            if dataset != trained_on:
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
    for metrics_path in sorted(Path(args.results_dir).glob("*/*/*/fold_*/*/*/metrics.csv")):
        run_dir = metrics_path.parents[2]
        model, probe, trained_on = _read_run(run_dir)
        fold = run_dir.name.removeprefix("fold_")
        tested_on = metrics_path.parent.name
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

        for suffix, statistic in statistics.items():
            bodies[("main", suffix)] += (
                f"<section><h1>{html.escape(family)}</h1>"
                + _render_table(main_records, family_datasets, statistic)
                + "</section>"
            )
            if ablation_records:
                bodies[("ablation", suffix)] += (
                    f"<section><h1>{html.escape(family)} — Ablation</h1>"
                    + _render_table(ablation_records, family_datasets, statistic)
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
