"""How far a finished model's score moves when the same section arrives flipped, turned or rescaled.

The main table cannot answer this. A run that scores 0.85 on its test set but collapses when the
image is mirrored has learned the orientation of its training set rather than the anatomy, and that
is the case for training with augmentation. Nothing here trains: it reads a checkpoint and evaluates
it under transforms it never saw.

Every score is measured in the case's *own* frame, at its own resolution, against the label the
annotator drew -- the transform is applied to the input, and the prediction is carried back through
the inverse before it is written. So these numbers sit directly beside the ones in the main report,
and the `none` row must reproduce them.

Unlike `data._augment`, a rotation here is not cropped back to real image. Cropping is a training
concern: a model must not be fitted on invented black. A section that genuinely arrives rotated does
have black corners, and cropping them away would change the field of view instead of measuring
whether the model minds.

Results land outside `models/` because `report.py` keys its table on the dataset a column was
evaluated on, whatever directory it came from, and a robustness column would silently overwrite the
ordinary one.

    PYTHONPATH=src python -m fm_adaptation.robustness \
        --config configs/dinov3_upernet_inj_ft_balanced_sci208.yaml
"""

import argparse
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

from .compute_metrics import _nanmean, measure
from .config import ExperimentConfig
from .data import (
    STAIN_PLANES,
    NnUNet2DDataset,
    collate_cases,
    load_dataset_json,
    num_classes,
    trained_planes,
)
from .metrics import read_case_metrics
from .models import load_trained_model, restore_prediction
from .predict import _has_labels, _jobs, _seen_in_training, _write_source


@dataclass(frozen=True)
class Transform:
    name: str
    hflip: bool = False
    vflip: bool = False
    degrees: float = 0.0
    scale: float = 1.0
    drop_smi: bool = False

    @property
    def directory(self) -> str:
        return self.name.replace(", ", "+").replace(" ", "_")


# Singles first, so a combination that hurts can be read against each of its parts.
TRANSFORMS = (
    Transform("none"),
    Transform("h-flip", hflip=True),
    Transform("v-flip", vflip=True),
    Transform("rotate +10", degrees=10.0),
    Transform("rotate -10", degrees=-10.0),
    Transform("scale 1.25", scale=1.25),
    Transform("scale 0.75", scale=0.75),
    Transform("drop SMI", drop_smi=True),
    Transform("h-flip, rotate +10", hflip=True, degrees=10.0),
    Transform("v-flip, scale 0.75", vflip=True, scale=0.75),
    Transform("rotate -10, scale 1.25", degrees=-10.0, scale=1.25),
    Transform("all, rotate +10, scale 0.75", hflip=True, vflip=True, degrees=10.0, scale=0.75),
)


def apply(images, transform, fill):
    """The transform, on a batch of preprocessed canvases."""
    if transform.drop_smi:
        images = images.clone()
        images[:, STAIN_PLANES["SMI"]] = fill[STAIN_PLANES["SMI"]]
    if transform.hflip:
        images = torch.flip(images, [-1])
    if transform.vflip:
        images = torch.flip(images, [-2])
    if (transform.degrees, transform.scale) != (0.0, 1.0):
        images = TF.affine(
            images, angle=transform.degrees, translate=[0, 0], scale=transform.scale,
            shear=[0.0, 0.0], interpolation=TF.InterpolationMode.BILINEAR, fill=list(fill),
        )
    return images


def invert(predictions, transform):
    """A batch of predicted label maps carried back into the frame the case was read in.

    Undone in the reverse of the order it was applied. Dropping a stain has no geometry to undo, and
    what a zoom or rotation pushed off the canvas cannot come back -- it returns as background, which
    is the honest cost of the transform rather than an error to paper over.
    """
    if (transform.degrees, transform.scale) != (0.0, 1.0):
        predictions = TF.affine(
            predictions.unsqueeze(1).float(), angle=-transform.degrees, translate=[0, 0],
            scale=1.0 / transform.scale, shear=[0.0, 0.0],
            interpolation=TF.InterpolationMode.NEAREST, fill=[0.0],
        ).squeeze(1).long()
    if transform.vflip:
        predictions = torch.flip(predictions, [-2])
    if transform.hflip:
        predictions = torch.flip(predictions, [-1])
    return predictions


def smi_cases(dataset):
    """How many of a dataset's selected cases actually carry SMI, so `drop SMI` can say what it drops.

    Eric's sections ship a blank SMI file, and a model trained on GFAP alone is never handed the
    plane at all; either way the transform can move nothing, and a table that did not say so would
    read as robustness. Counted from the files rather than from a batch, so it is the same number
    whether or not the predictions were cached.
    """
    planes = dataset.stain_planes or {}
    if "SMI" not in planes:
        return 0
    stored = planes["SMI"][0]
    image_dir = dataset.dataset_dir / f"images{dataset.split}"
    count = 0
    for case_id in dataset.ids:
        probe = cv2.imread(
            str(image_dir / f"{case_id}_{stored:04d}{dataset.ending}"), cv2.IMREAD_REDUCED_GRAYSCALE_8
        )
        if probe is not None and probe.any():
            count += 1
    return count


def encoder_black(preprocess):
    """What black becomes once the encoder has normalised it -- the value its padding already holds."""
    return preprocess(np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.uint8))[0][:, 0, 0].tolist()


def evaluate(model, loader, transform, device, amp, fill, output_dir, description):
    """Predict one column under one transform, writing PNGs in the cases' own resolution."""
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for images, _, metadata in tqdm(loader, desc=description, leave=False):
            with amp:
                logits = model(apply(images, transform, fill).to(device, non_blocking=True))
            for prediction, meta in zip(invert(logits.argmax(1).cpu(), transform), metadata):
                cv2.imwrite(
                    str(prediction_dir / f"{meta['case_id']}.png"),
                    restore_prediction(prediction, meta),
                )
    return prediction_dir


def _cases(loaders):
    return sum(len(loader.dataset.ids) for loader in loaders)


def _mean_dice(metrics_path):
    """A column's dice, averaged over its cases.

    `read_case_metrics` drops the file's own MEAN row -- it is not a case -- and reports a case whose
    label has no foreground as NaN, so averaging what is left reproduces that row rather than
    inventing a second way of computing it.
    """
    if not metrics_path.exists():
        return None
    score = _nanmean(row["dice"] for row in read_case_metrics(metrics_path))
    return None if score != score else score


def _short(column):
    """`Dataset214_lesion_mohammad_smi_gfap` is a heading nobody can read."""
    parts = column.split("_")
    if len(parts) < 4 or not parts[0].startswith("Dataset"):
        return column
    return f"{parts[0].removeprefix('Dataset')} {'_'.join(parts[2:-2])}"


def _cell(score):
    return "--" if score is None else f"{score:.3f}"


def table(rows, columns, smi_counts, run_label):
    """Mean dice per transform, then the same as a change against `none`."""
    heading = "| transform | " + " | ".join(_short(c) for c in columns) + " |"
    divider = "| --- |" + " --- |" * len(columns)
    lines = [
        f"# Robustness of {run_label}", "",
        "Mean dice, measured at each case's own resolution against its own label, so these sit "
        "beside the main table. Rotation and scale are applied to the input and undone on the "
        "prediction; nothing is cropped away first.", "",
        heading, divider,
    ]
    for name, scores in rows:
        lines.append(f"| {name} | " + " | ".join(_cell(scores.get(c)) for c in columns) + " |")

    reference = dict(rows).get("none", {})
    lines += ["", "Change against `none`.", "", heading, divider]
    for name, scores in rows:
        if name == "none":
            continue
        deltas = []
        for column in columns:
            score, base = scores.get(column), reference.get(column)
            deltas.append("--" if score is None or base is None else f"{score - base:+.3f}")
        lines.append(f"| {name} | " + " | ".join(deltas) + " |")

    lines += [
        "", "Cases carrying SMI, of those evaluated. `drop SMI` can only move a column that has "
        "some -- Eric's sections ship a blank SMI file.", "",
        "| " + " | ".join(_short(c) for c in columns) + " |", "|" + " --- |" * len(columns),
        "| " + " | ".join(f"{smi_counts[c][0]}/{smi_counts[c][1]}" for c in columns) + " |",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", choices=("best", "final", "last"), default="final")
    parser.add_argument("--output-dir", default="results_analysis/robustness")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-predict transforms that already have predictions")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device(cfg.device)
    classes = num_classes(cfg.raw_data_dir / cfg.train_dataset)
    model = load_trained_model(cfg, args.checkpoint, device, classes)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    keep_planes = trained_planes(
        load_dataset_json(cfg.raw_data_dir / cfg.train_dataset)["channel_names"], cfg.stains
    )
    fill = encoder_black(model.encoder.preprocess)
    seen = _seen_in_training(cfg)

    # The same jobs `predict.py` runs, so the cases are the cases the main table reports. A column
    # can be reached by more than one job -- `test_split: all` evaluates a set that ships both
    # `imagesTr` and `imagesTs` whole -- and each keeps its own loader, since a split is a directory
    # and not just a list of ids. They write into one directory, which is what makes them one column.
    jobs, columns = {}, []
    selected = list(_jobs(cfg, cfg.test_datasets or (cfg.train_dataset,)))
    # The training set is reported once: on its held-out `imagesTs` where there is one, otherwise on
    # the fold's validation split. Scoring both into one column would mix them.
    if any(kind == "test" and column == cfg.train_dataset for _, _, _, kind, column in selected):
        selected = [job for job in selected if not (job[3] == "validation" and job[4] == cfg.train_dataset)]
    for dataset_name, split, subset, kind, column in selected:
        if not _has_labels(cfg.raw_data_dir / dataset_name, split):
            continue  # Dataset212 ships images only; a transform has nothing to be scored against.
        dataset = NnUNet2DDataset(
            cfg.raw_data_dir, dataset_name, split, cfg.fold, subset, model.encoder.preprocess,
            keep_planes=keep_planes, require_labels=True,
        )
        if dataset_name != cfg.train_dataset:
            dataset.ids = [case for case in dataset.ids if case not in seen]
        if not dataset.ids:
            continue
        if column not in jobs:
            jobs[column] = []
            columns.append(column)
        jobs[column].append(DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            collate_fn=collate_cases, pin_memory=True,
        ))

    root = Path(args.output_dir)
    fold_dir = root / cfg.model_name / cfg.train_dataset / cfg.run_name / f"fold_{cfg.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    # `plot_qualitative` reads the run's config from the fold directory it finds a prediction under,
    # so the figures come from the same command rather than needing the tree rebuilt elsewhere.
    shutil.copyfile(cfg.run_dir / "config.yaml", fold_dir / "config.yaml")

    rows = []
    for transform in TRANSFORMS:
        scores = {}
        for column in columns:
            output_dir = fold_dir / transform.directory / column
            prediction_dir = output_dir / "predictions"
            # `source.json` is written only once every loader for the column has finished, so a run
            # killed mid-column is redone rather than trusted: the directory existing proves only
            # that something started writing into it.
            if args.overwrite or not (output_dir / "source.json").exists():
                for loader in jobs[column]:
                    evaluate(
                        model, loader, transform, device, amp, fill, output_dir,
                        f"{transform.name} | {column}",
                    )
                _write_source(output_dir, column, cfg.test_split)
            print(measure(prediction_dir, cfg.raw_data_dir, overwrite=args.overwrite))
            scores[column] = _mean_dice(output_dir / "metrics.csv")
        rows.append((transform.name, scores))
        print(f"{transform.name:32s} " + "  ".join(_cell(scores.get(c)) for c in columns))

    smi_counts = {
        column: (sum(smi_cases(loader.dataset) for loader in loaders), _cases(loaders))
        for column, loaders in jobs.items()
    }
    label = f"{cfg.model_name} {cfg.run_name} on {cfg.train_dataset} fold {cfg.fold}"
    path = root / f"{cfg.model_name}__{cfg.train_dataset}__{cfg.run_name}__fold_{cfg.fold}.md"
    rendered = table(rows, columns, smi_counts, label)
    path.write_text(rendered)
    print("\n" + rendered + f"\nwrote {path}")


if __name__ == "__main__":
    main()
