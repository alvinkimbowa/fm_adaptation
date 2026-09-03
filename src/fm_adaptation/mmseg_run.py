"""Train and evaluate the DINOv3 adapter heads through mmsegmentation's own Runner.

Reads the same YAML configs as the rest of the repo and writes results into the same
`models/<model>/<dataset>/<run>/fold_<n>/` layout, so `report.py` and the figure scripts need no
changes. Everything between those two ends — transforms, losses, optimiser, training loop, sliding-window
inference — is stock mmseg.
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from .config import ExperimentConfig
from .datasets import dataset_dir as resolve_dataset_dir
from .data import _case_ids, load_dataset_json, num_classes
from .dinov3_mmseg import head_cfg


def _pipelines(cfg, crop_size, ending):
    """Random crops for patchwise datasets, whole-image resize otherwise."""
    load = [
        {"type": "LoadImageFromFile"},
        {"type": "LoadAnnotations", "reduce_zero_label": False},
    ]
    if cfg.patching is not None:
        train = load + [
            {"type": "RandomResize", "scale": (crop_size * 2, crop_size * 2), "ratio_range": (0.75, 1.25),
             "keep_ratio": True},
            {"type": "RandomCrop", "crop_size": (crop_size, crop_size), "cat_max_ratio": 0.9},
            {"type": "RandomFlip", "prob": 0.5},
            {"type": "PhotoMetricDistortion"},
            {"type": "PackSegInputs"},
        ]
    else:
        train = load + [
            {"type": "Resize", "scale": (crop_size, crop_size), "keep_ratio": False},
            {"type": "RandomFlip", "prob": 0.5},
            {"type": "PhotoMetricDistortion"},
            {"type": "PackSegInputs"},
        ]
    test = [{"type": "LoadImageFromFile"}]
    if cfg.patching is None:
        test.append({"type": "Resize", "scale": (crop_size, crop_size), "keep_ratio": False})
    test += [{"type": "LoadAnnotations", "reduce_zero_label": False}, {"type": "PackSegInputs"}]
    return train, test


def _dataset(cfg, dataset_name, split, subset, pipeline, ending):
    dataset_dir = cfg.raw_data_dir / dataset_name
    return {
        "type": "NnUNetSegDataset",
        "case_ids": _case_ids(dataset_dir, split, cfg.fold, subset),
        "ext": ending,
        "data_root": str(dataset_dir),
        "data_prefix": {"img_path": f"images{split}", "seg_map_path": f"labels{split}"},
        "pipeline": pipeline,
    }


def _runner_cfg(cfg, head, crop_size, work_dir, epochs, batch_size, lr, amp_dtype, injector=False):
    dataset_dir = cfg.raw_data_dir / cfg.train_dataset
    ending = load_dataset_json(dataset_dir)["file_ending"]
    classes = num_classes(dataset_dir)
    train_pipeline, test_pipeline = _pipelines(cfg, crop_size, ending)
    n_train = len(_case_ids(dataset_dir, "Tr", cfg.fold, "train"))
    warmup = int(min(500, max(50, n_train // batch_size)))
    model = head_cfg(head, classes, crop_size, injector=injector)
    if cfg.patching is not None:
        stride = int(crop_size * (1 - cfg.patching.overlap))
        model["test_cfg"] = {"mode": "slide", "crop_size": (crop_size, crop_size), "stride": (stride, stride)}

    return Config(
        {
            "model": model,
            "default_scope": "mmseg",
            "work_dir": str(work_dir),
            "randomness": {"seed": cfg.seed},
            "env_cfg": {"cudnn_benchmark": True, "mp_cfg": {"mp_start_method": "fork"}, "dist_cfg": {"backend": "nccl"}},
            "log_processor": {"by_epoch": True},
            "default_hooks": {
                "timer": {"type": "IterTimerHook"},
                "logger": {"type": "LoggerHook", "interval": 10, "log_metric_by_epoch": True},
                "param_scheduler": {"type": "ParamSchedulerHook"},
                "checkpoint": {"type": "CheckpointHook", "by_epoch": True, "interval": epochs, "save_last": True,
                               "max_keep_ckpts": 1},
                "sampler_seed": {"type": "DistSamplerSeedHook"},
            },
            "train_dataloader": {
                "batch_size": batch_size,
                "num_workers": cfg.num_workers,
                "persistent_workers": True,
                # An odd case count leaves a batch of 1, which UPerHead's 1x1 pooling branch cannot
                # BatchNorm; dropping the remainder is the standard mmseg behaviour.
                "drop_last": True,
                "sampler": {"type": "DefaultSampler", "shuffle": True},
                "dataset": _dataset(cfg, cfg.train_dataset, "Tr", "train", train_pipeline, ending),
            },
            "val_dataloader": {
                "batch_size": 1,
                "num_workers": cfg.num_workers,
                "persistent_workers": True,
                "sampler": {"type": "DefaultSampler", "shuffle": False},
                "dataset": _dataset(cfg, cfg.train_dataset, "Tr", "val", test_pipeline, ending),
            },
            "val_evaluator": {"type": "IoUMetric", "iou_metrics": ["mDice"]},
            "train_cfg": {"type": "EpochBasedTrainLoop", "max_epochs": epochs, "val_interval": max(1, epochs // 10)},
            "val_cfg": {"type": "ValLoop"},
            "optim_wrapper": {
                # Mask2Former runs in fp32: its deformable-attention kernel has no bfloat16 path, and
                # under float16 the mask/dice costs overflow to NaN, which makes the Hungarian
                # assigner raise "cost matrix is infeasible". fp32 peaks at 8.7 GB, so it fits.
                "type": "OptimWrapper" if amp_dtype is None else "AmpOptimWrapper",
                **({} if amp_dtype is None else {"dtype": amp_dtype}),
                "optimizer": {"type": "AdamW", "lr": lr, "weight_decay": cfg.weight_decay},
                "clip_grad": {"max_norm": 1.0, "norm_type": 2},
            },
            "param_scheduler": [
                # Roughly one epoch of warmup. A fixed 50 iterations is far too short on the larger
                # datasets, where Mask2Former's early gradients overflow fp16 and hand the Hungarian
                # assigner a NaN cost matrix.
                {"type": "LinearLR", "start_factor": 1e-3, "by_epoch": False, "begin": 0, "end": warmup},
                {"type": "PolyLR", "eta_min": 0.0, "power": 1.0, "begin": 0, "end": epochs, "by_epoch": True},
            ],
        }
    )


def _predict(runner, cfg, test_pipeline, run_dir):
    """Per-case inference through the trained model, written at each case's native resolution.

    Which datasets and splits an evaluation covers comes from `predict._jobs`, so a run trained here
    is evaluated on exactly what the same config would be evaluated on through the repo's own loop.
    Scoring is a separate stage: `fm_adaptation.compute_metrics` reads these predictions off disk.
    """
    import cv2
    from mmengine.dataset import Compose, default_collate

    from .patching import open_image
    from .predict import _jobs, _seen_in_training

    model = runner.model
    model.eval()
    # LoadAnnotations would need the label at inference, which nothing here reads.
    pipeline = Compose([step for step in test_pipeline if step["type"] != "LoadAnnotations"])
    datasets = cfg.test_datasets or (cfg.train_dataset,)

    for dataset_name, split, subset, kind, column in _jobs(cfg, datasets, _seen_in_training(cfg)):
        dataset_dir = cfg.raw_data_dir / dataset_name
        ending = load_dataset_json(dataset_dir)["file_ending"]
        case_ids = _case_ids(dataset_dir, split, cfg.fold, subset)
        output_dir = run_dir / kind / column / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        for case_id in tqdm(case_ids, desc=f"{kind} {column} images{split}"):
            image_path = dataset_dir / f"images{split}" / f"{case_id}_0000{ending}"
            data = pipeline({"img_path": str(image_path), "seg_fields": [], "reduce_zero_label": False})
            with torch.no_grad():
                result = model.test_step(default_collate([data]))[0]
            prediction = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
            height, width = open_image(image_path).shape[:2]
            if prediction.shape != (height, width):
                prediction = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(output_dir / f"{case_id}.png"), prediction)
        print(f"Wrote {len(case_ids)} predictions to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--head", choices=("upernet", "m2f"), required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, default=896)
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--injector", type=int, default=0,
                        help="restore ViT-Adapter's injector, so the adapter writes back into the ViT")
    args = parser.parse_args()

    init_default_scope("mmseg")
    import mmdet.models  # noqa: F401  registers Mask2Former pieces
    import mmseg.models  # noqa: F401

    from .dinov3_mmseg import _dataset_class

    _dataset_class()  # registers NnUNetSegDataset

    cfg = ExperimentConfig.from_yaml(args.config)
    # The injector variant is a separate run so it sits beside the extractor-only one in the tables
    # instead of overwriting it.
    run_name = f"{args.head}_inj" if args.injector else args.head
    run_dir = cfg.results_dir / cfg.model_name / cfg.train_dataset / run_name / f"fold_{cfg.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # report.py identifies a run from its config, so the head has to be recorded there; copying the
    # probe config verbatim would make every head claim to be the linear probe.
    _write_run_config(args.config, run_name, run_dir / "config.yaml")
    amp_dtype = None if args.head == "m2f" else "bfloat16"

    runner_cfg = _runner_cfg(cfg, args.head, args.crop_size, run_dir / "mm", args.epochs,
                             args.batch_size, args.lr, amp_dtype, injector=bool(args.injector))
    runner = Runner.from_cfg(runner_cfg)
    if args.train:
        runner.train()
        _write_history(run_dir)
        _save_trainable(runner.model, run_dir / "final.pt")
        # mmengine's own checkpoints include the frozen backbone (~1.8 GB each); final.pt supersedes them.
        for checkpoint in (run_dir / "mm").glob("*.pth"):
            checkpoint.unlink()
    else:
        _load_trainable(runner.model, run_dir / "final.pt")

    ending = load_dataset_json(cfg.raw_data_dir / cfg.train_dataset)["file_ending"]
    _, test_pipeline = _pipelines(cfg, args.crop_size, ending)
    _predict(runner, cfg, test_pipeline, run_dir)


def _write_run_config(source, run_name, destination):
    import yaml

    with open(source) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["run_name"] = run_name
    with open(destination, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _save_trainable(model, path):
    """The DINOv3 backbone is frozen and reloaded from its own checkpoint, so only store what trains."""
    state = {k: v for k, v in model.state_dict().items() if "adapter.backbone" not in k}
    torch.save({"state_dict": state}, path)
    print(f"saved {len(state)} trainable tensors to {path} ({path.stat().st_size / 1e6:.0f} MB)")


def _load_trainable(model, path):
    """Restore what `_save_trainable` kept. Predicting without training would otherwise run a freshly
    initialised adapter and head, so a missing checkpoint has to be an error rather than a warning."""
    if not path.exists():
        raise SystemExit(f"No trained weights at {path}; run with --train 1 first")
    state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The frozen DINOv3 trunk is rebuilt from its own checkpoint, so it is legitimately absent here.
    missing = [key for key in missing if "adapter.backbone" not in key]
    if missing or unexpected:
        raise RuntimeError(f"{path}: missing={missing[:5]} unexpected={unexpected[:5]}")
    print(f"loaded {len(state)} trainable tensors from {path}")


def _write_history(run_dir):
    """Translate mmengine's scalar log into the repo's history.csv so plot_history works unchanged."""
    logs = sorted((run_dir / "mm").glob("*/vis_data/scalars.json"))
    if not logs:
        return
    import json

    rows = {}
    for line in logs[-1].read_text().splitlines():
        entry = json.loads(line)
        if "mDice" in entry:
            # Validation entries carry the epoch as `step`, not `epoch`.
            rows.setdefault(entry["step"], {})["val_dice"] = entry["mDice"] / 100.0
        elif "loss" in entry and "epoch" in entry:
            rows.setdefault(entry["epoch"], {}).setdefault("losses", []).append(entry["loss"])
    for row in rows.values():
        if "losses" in row:
            row["train_loss"] = sum(row["losses"]) / len(row["losses"])
    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice"])
        for epoch in sorted(rows):
            row = rows[epoch]
            writer.writerow([epoch, row.get("train_loss", ""), "", "", row.get("val_dice", "")])


if __name__ == "__main__":
    main()
