from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PatchConfig:
    """Patchwise training and inference on images too large to resize to the encoder input."""

    patch_size: int = 1008
    patches_per_case: int = 16
    min_roi_fraction: float = 0.05
    roi_threshold: int = 0
    overlap: float = 0.5
    ignore_masked_out: bool = False

    @property
    def stride(self) -> int:
        return max(1, round(self.patch_size * (1.0 - self.overlap)))


@dataclass(frozen=True)
class ExperimentConfig:
    raw_data_dir: Path
    results_dir: Path
    train_dataset: str
    test_datasets: tuple[str, ...]
    test_split: str
    fold: str
    model_name: str
    checkpoint: str | None
    variant: str
    probe_name: str
    injector: bool
    train_encoder: bool
    init_from: str | None
    init_from_checkpoint: str
    run_name: str
    epochs: int
    min_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    batch_size: int
    encoder_batch_size: int
    num_workers: int
    learning_rate: float
    encoder_learning_rate: float | None
    encoder_layer_decay: float
    lr_schedule: str
    lr_warmup_iters: int | None
    lr_warmup_start_factor: float
    lr_power: float
    accumulation_steps: int
    weight_decay: float
    seed: int
    device: str
    patching: "PatchConfig | None"
    channel_dropout: tuple[str, ...]
    channel_dropout_p: float
    balance_sources: bool

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        data = cfg["data"]
        model = cfg["model"]
        training = cfg["training"]
        patching = cfg.get("patching") or {}
        channel_dropout_p = float(data.get("channel_dropout_p", 0.5))
        if not 0.0 <= channel_dropout_p <= 1.0:
            raise ValueError("data.channel_dropout_p must be between 0 and 1")
        return cls(
            raw_data_dir=Path(data["raw_data_dir"]),
            results_dir=Path(data.get("results_dir", "models")),
            train_dataset=str(data["train_dataset"]),
            test_datasets=tuple(str(x) for x in data.get("test_datasets", [])),
            test_split=str(data.get("test_split", "Tr")),
            fold=str(data["fold"]),
            model_name=str(model["name"]),
            checkpoint=model.get("checkpoint"),
            # Which DINOv3 trunk size to build; the default is the ViT-L every existing run used.
            variant=str(model.get("variant", "vitl16")),
            probe_name=str(model["probe"]),
            # Only meaningful for the adapter-based decoders; ignored by the probes.
            injector=bool(model.get("injector", False)),
            # Unfreeze the foundation-model trunk and train it along with the adapter and the head.
            train_encoder=bool(model.get("train_encoder", False)),
            # Start from another run's weights instead of a fresh head, the way `LP + FT` starts from
            # its own probe: the run name to seed from, within the same model, dataset and fold.
            init_from=model.get("init_from"),
            init_from_checkpoint=str(model.get("init_from_checkpoint", "final")),
            run_name=str(model.get("run_name", model["probe"])),
            epochs=int(training["epochs"]),
            min_epochs=int(training.get("min_epochs", 1)),
            early_stopping_patience=int(training.get("early_stopping_patience", 0)),
            early_stopping_min_delta=float(training.get("early_stopping_min_delta", 0.0)),
            batch_size=int(training["batch_size"]),
            encoder_batch_size=int(training.get("encoder_batch_size", 4)),
            num_workers=int(training.get("num_workers", 4)),
            learning_rate=float(training["learning_rate"]),
            # A pretrained trunk cannot take the head's learning rate; when it trains it gets its own.
            encoder_learning_rate=(
                float(training["encoder_learning_rate"])
                if training.get("encoder_learning_rate") is not None
                else None
            ),
            # Layer-wise decay of that rate down the trunk, as ViT-Adapter does; 1.0 is a flat rate.
            encoder_layer_decay=float(training.get("encoder_layer_decay", 1.0)),
            # `poly` is ViT-Adapter's schedule: a linear warmup, then a decay to zero over the run.
            # `none`, the default, is the constant rate every run before this one trained at.
            lr_schedule=str(training.get("lr_schedule", "none")),
            # Length of the warmup in optimiser steps; unset is roughly one epoch, clamped to [50, 500].
            lr_warmup_iters=(
                int(training["lr_warmup_iters"])
                if training.get("lr_warmup_iters") is not None
                else None
            ),
            # A multiplier on each group's own rate, not a rate: the first step runs at base * this.
            lr_warmup_start_factor=float(training.get("lr_warmup_start_factor", 1e-3)),
            # Exponent of the poly decay; 1.0 is a straight line to zero.
            lr_power=float(training.get("lr_power", 1.0)),
            accumulation_steps=int(training.get("accumulation_steps", 1)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            seed=int(training.get("seed", 0)),
            device=str(training.get("device", "cuda")),
            channel_dropout=tuple(str(x).upper() for x in data.get("channel_dropout", [])),
            channel_dropout_p=channel_dropout_p,
            balance_sources=bool(data.get("balance_sources", False)),
            patching=(
                PatchConfig(
                    patch_size=int(patching.get("patch_size", 1008)),
                    patches_per_case=int(patching.get("patches_per_case", 16)),
                    min_roi_fraction=float(patching.get("min_roi_fraction", 0.05)),
                    roi_threshold=int(patching.get("roi_threshold", 0)),
                    overlap=float(patching.get("overlap", 0.5)),
                    ignore_masked_out=bool(patching.get("ignore_masked_out", False)),
                )
                if patching.get("enabled")
                else None
            ),
        )

    @property
    def run_dir(self) -> Path:
        return (
            self.results_dir
            / self.model_name
            / self.train_dataset
            / self.run_name
            / f"fold_{self.fold}"
        )

    @property
    def feature_cache_dir(self) -> Path:
        return self.results_dir / ".feature_cache" / self.model_name / self.train_dataset

    def patch_cache_dir(self, dataset_name: str) -> Path:
        return self.results_dir / ".patch_cache" / dataset_name
