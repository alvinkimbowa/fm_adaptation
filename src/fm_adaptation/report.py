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


def _mean_sd(values, scale=1.0):
    values = values[np.isfinite(values)] * scale
    ddof = 1 if len(values) > 1 else 0
    return f"{values.mean():.2f} ± {values.std(ddof=ddof):.2f}"


def _median_iqr(values, scale=1.0):
    values = values[np.isfinite(values)] * scale
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return f"{median:.2f} ({q1:.2f}–{q3:.2f})"


def _render_table(records, datasets, statistic):
    fmt = _mean_sd if statistic == "Mean ± SD" else _median_iqr
    parts = [f"<h2>{html.escape(statistic)}</h2><table><thead><tr>"]
    for heading in ("Config", "Trained on", "Fold"):
        parts.append(f"<th rowspan='2'>{heading}</th>")
    for dataset in datasets:
        parts.append(f"<th colspan='2'>{html.escape(dataset)}</th>")
    parts.append("<th colspan='2'>Cross-dataset average</th></tr><tr>")
    parts.extend("<th>Dice ↑</th><th>MASD (px) ↓</th>" for _ in range(len(datasets) + 1))
    parts.append("</tr></thead><tbody>")
    for key, results in sorted(records.items()):
        model, probe, trained_on, fold = key
        parts.append(
            f"<tr><td>{html.escape(model.upper() + ' + ' + probe.upper())}</td>"
            f"<td>{html.escape(trained_on)}</td><td>{html.escape(fold)}</td>"
        )
        cross_dice, cross_masd = [], []
        for dataset in datasets:
            metrics = results.get(dataset)
            if metrics is None:
                parts.append("<td>—</td><td>—</td>")
                continue
            parts.append(f"<td>{fmt(metrics['dice'], 100)}</td><td>{fmt(metrics['masd'])}</td>")
            if dataset != trained_on:
                reducer = np.mean if statistic == "Mean ± SD" else np.median
                cross_dice.append(reducer(metrics["dice"]))
                cross_masd.append(reducer(metrics["masd"]))
        if cross_dice:
            parts.append(
                f"<td>{fmt(np.asarray(cross_dice), 100)}</td>"
                f"<td>{fmt(np.asarray(cross_masd))}</td>"
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
        for (model, adaptation, trained_on, fold), results in sorted(records.items()):
            for tested_on, metrics in sorted(results.items()):
                dice, masd = metrics["dice"], metrics["masd"]
                writer.writerow(
                    [
                        model,
                        adaptation,
                        trained_on,
                        fold,
                        tested_on,
                        len(dice),
                        np.mean(dice),
                        np.std(dice, ddof=1),
                        np.median(dice),
                        *np.percentile(dice, [25, 75]),
                        np.mean(masd),
                        np.std(masd, ddof=1),
                        np.median(masd),
                        *np.percentile(masd, [25, 75]),
                    ]
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="models")
    parser.add_argument("--output", default="models/cross_dataset_report.html")
    args = parser.parse_args()
    records = defaultdict(dict)
    datasets = set()
    for metrics_path in Path(args.results_dir).glob("*/*/*/fold_*/*/*/metrics.csv"):
        run_dir = metrics_path.parents[2]
        model, probe, trained_on = _read_run(run_dir)
        fold = run_dir.name.removeprefix("fold_")
        tested_on = metrics_path.parent.name
        records[(model, probe, trained_on, fold)][tested_on] = _read_metrics(metrics_path)
        datasets.add(tested_on)
    if not records:
        raise RuntimeError(f"No cross-dataset metrics found under {args.results_dir}")
    style = """
    body{background:#111;color:#bbb;font-family:system-ui;margin:16px}table{border-collapse:collapse;width:100%}
    th,td{padding:7px 10px;text-align:right}th{background:#292929}td{background:#191919;border-bottom:1px solid #222}
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}h2{font-size:16px;font-weight:400;margin-top:28px}
    """
    ordered_datasets = sorted(datasets)
    body = _render_table(records, ordered_datasets, "Mean ± SD")
    body += _render_table(records, ordered_datasets, "Median (Q1–Q3)")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"<!doctype html><meta charset='utf-8'><style>{style}</style>{body}")
    _write_summary_csv(records, output.with_suffix(".csv"))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
