import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


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
        records[("nnU-Net", "", trained_on, fold)][tested_on] = _read_metrics(
            metrics_path
        )


def _mean_sd(values, scale=1.0):
    values = values[~np.isnan(values)] * scale
    if np.isinf(values).any():
        return "∞"
    ddof = 1 if len(values) > 1 else 0
    return f"{values.mean():.2f} ± {values.std(ddof=ddof):.2f}"


def _median_iqr(values, scale=1.0):
    values = values[~np.isnan(values)] * scale
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    fmt = lambda value: "∞" if np.isinf(value) else f"{value:.2f}"
    return f"{fmt(median)} ({fmt(q1)}–{fmt(q3)})"


def _dataset_family(dataset):
    return dataset.split("_", maxsplit=2)[1]


def _config_label(model, adaptation):
    report_model, report_adaptation = _report_names(model, adaptation)
    return " + ".join(value for value in (report_model, report_adaptation) if value)


ADAPTATIONS = {
    "linear": ("LP", 0),
    "nonlinear": ("NLP", 1),
    "linear_finetune": ("LP + FT", 2),
    "nonlinear_finetune": ("NLP + FT", 3),
    "": ("", 4),
}


def _split_adaptation(adaptation):
    """Split a run name into its known base and any sweep suffix (e.g. `_wd1.0`)."""
    for base in sorted(ADAPTATIONS, key=len, reverse=True):
        if adaptation == base:
            return base, ""
        if base and adaptation.startswith(f"{base}_"):
            return base, adaptation[len(base) + 1 :]
    return "", adaptation


def _report_names(model, adaptation):
    if model == "nnU-Net":
        return model, ""
    models = {"sam3": "SAM3"}
    base, suffix = _split_adaptation(adaptation)
    label = ADAPTATIONS[base][0]
    if suffix:
        label = f"{label} ({suffix})" if label else suffix
    return models[model], label


def _experiment_order(item):
    model, adaptation, trained_on, fold = item[0]
    base, suffix = _split_adaptation(adaptation)
    return trained_on, ADAPTATIONS[base][1], suffix, fold, model


def _best_values(records, datasets, reducer):
    best = {}
    for (_, _, trained_on, _), results in records.items():
        cross = {"dice": [], "masd": []}
        for dataset in datasets:
            metrics = results.get(dataset)
            if metrics is None:
                continue
            for metric in ("dice", "masd"):
                value = reducer(metrics[metric])
                key = trained_on, dataset, metric
                choose = max if metric == "dice" else min
                best[key] = choose(best.get(key, value), value)
                if dataset != trained_on:
                    cross[metric].append(value)
        for metric, values in cross.items():
            if not values:
                continue
            value = reducer(values)
            key = trained_on, "cross", metric
            choose = max if metric == "dice" else min
            best[key] = choose(best.get(key, value), value)
    return best


def _metric_cell(text, value, best):
    if np.isclose(value, best, equal_nan=False):
        text = f"<strong>{text}</strong>"
    return f"<td>{text}</td>"


def _render_table(records, datasets, statistic):
    fmt = _mean_sd if statistic == "Mean ± SD" else _median_iqr
    reducer = np.mean if statistic == "Mean ± SD" else np.median
    best = _best_values(records, datasets, reducer)
    parts = [f"<h2>{html.escape(statistic)}</h2><table><thead><tr>"]
    for heading in ("Config", "Trained on", "Fold"):
        parts.append(f"<th rowspan='2'>{heading}</th>")
    for dataset in datasets:
        parts.append(f"<th colspan='2'>{html.escape(dataset)}</th>")
    parts.append("<th colspan='2'>Cross-dataset average</th></tr><tr>")
    parts.extend("<th>Dice ↑</th><th>MASD (px) ↓</th>" for _ in range(len(datasets) + 1))
    parts.append("</tr></thead><tbody>")
    previous_trained_on = None
    for key, results in sorted(records.items(), key=_experiment_order):
        model, probe, trained_on, fold = key
        config = _config_label(model, probe)
        row_class = " class='group-start'" if previous_trained_on not in (None, trained_on) else ""
        parts.append(
            f"<tr{row_class}><td>{html.escape(config)}</td>"
            f"<td>{html.escape(trained_on)}</td><td>{html.escape(fold)}</td>"
        )
        previous_trained_on = trained_on
        cross_dice, cross_masd = [], []
        for dataset in datasets:
            metrics = results.get(dataset)
            if metrics is None:
                parts.append("<td>—</td><td>—</td>")
                continue
            dice_value = reducer(metrics["dice"])
            masd_value = reducer(metrics["masd"])
            parts.append(
                _metric_cell(
                    fmt(metrics["dice"], 100),
                    dice_value,
                    best[(trained_on, dataset, "dice")],
                )
            )
            parts.append(
                _metric_cell(
                    fmt(metrics["masd"]),
                    masd_value,
                    best[(trained_on, dataset, "masd")],
                )
            )
            if dataset != trained_on:
                cross_dice.append(dice_value)
                cross_masd.append(masd_value)
        if cross_dice:
            cross_dice_value = reducer(cross_dice)
            cross_masd_value = reducer(cross_masd)
            parts.append(
                _metric_cell(
                    fmt(np.asarray(cross_dice), 100),
                    cross_dice_value,
                    best[(trained_on, "cross", "dice")],
                )
            )
            parts.append(
                _metric_cell(
                    fmt(np.asarray(cross_masd)),
                    cross_masd_value,
                    best[(trained_on, "cross", "masd")],
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
    parser.add_argument("--nnunet-results-dir")
    parser.add_argument("--output", default="models/cross_dataset_report.html")
    args = parser.parse_args()
    records = defaultdict(dict)
    for metrics_path in Path(args.results_dir).glob("*/*/*/fold_*/*/*/metrics.csv"):
        run_dir = metrics_path.parents[2]
        model, probe, trained_on = _read_run(run_dir)
        fold = run_dir.name.removeprefix("fold_")
        tested_on = metrics_path.parent.name
        records[(model, probe, trained_on, fold)][tested_on] = _read_metrics(metrics_path)
    if args.nnunet_results_dir:
        _add_nnunet_records(records, args.nnunet_results_dir)
    if not records:
        raise RuntimeError(f"No cross-dataset metrics found under {args.results_dir}")
    style = """
    body{background:#111;color:#bbb;font-family:system-ui;margin:16px}table{border-collapse:collapse;width:100%}
    th,td{padding:7px 10px;text-align:right}th{background:#292929}td{background:#191919;border-bottom:1px solid #222}
    tr.group-start td{border-top:3px solid #777}strong{color:#eee;font-weight:700}
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
    section{margin-bottom:56px}h1{color:#ddd;font-size:22px;margin:0 0 18px}h2{font-size:16px;font-weight:400;margin-top:28px}
    """
    body = ""
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
        body += f"<section><h1>{html.escape(family)}</h1>"
        body += _render_table(family_records, family_datasets, "Mean ± SD")
        body += _render_table(family_records, family_datasets, "Median (Q1–Q3)")
        body += "</section>"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"<!doctype html><meta charset='utf-8'><style>{style}</style>{body}")
    _write_summary_csv(records, output.with_suffix(".csv"))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
