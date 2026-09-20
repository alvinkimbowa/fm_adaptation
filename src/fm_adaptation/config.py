from dataclasses import dataclass
from pathlib import Path

import yaml
from types import SimpleNamespace

from .datasets import resolve


@dataclass(frozen=True)
class PatchConfig:
    """Patchwise training and inference on images too large to resize to the encoder input."""

    patch_size: int = 1008
    patches_per_case: int = 16
    min_roi_fraction: float = 0.05
    roi_threshold: int = 0
    overlap: float = 0.5
    ignore_masked_out: bool = False
    # Compose known stain files into their declared RGB planes instead of replicating _0000.
    respect_channels: bool = False

    @property
    def stride(self) -> int:
        return max(1, round(self.patch_size * (1.0 - self.overlap)))


@dataclass(frozen=True)
class AugmentConfig:
    """Geometric augmentation of the training images: flips and a small rotation."""

    hflip: bool = True
    vflip: bool = True
    # Degrees; each case is rotated by an angle drawn uniformly from [-rotation, +rotation].
    rotation: float = 0.0
    flip_p: float = 0.5
    # Scale drawn uniformly from [zoom_min, zoom_max]; 1.0/1.0 is no zoom. A zoom that would carry
    # part of the annotation off the canvas is reduced until it fits -- see `_augment`.
    zoom_min: float = 1.0
    zoom_max: float = 1.0


@dataclass(frozen=True)
class PromptField:
    """One question the prompt asks, with the answers it accepts.

    `vocabulary` is what fixes each answer's index, so it is written down in the config rather than
    discovered from the data: the run directory keeps its config, and prediction reads the same order
    back. A value outside the vocabulary is dropped, and a field left with nothing takes its null.

    A `multi` field is answered with a set rather than one entry, and its vocabulary names classes of
    the training dataset: training draws a subset of the structures a frame actually has annotated
    and restricts the target to them, so the frames a structure was never traced on cannot teach that
    it is absent. `eval_values` pins the answer for validation and test, which is what makes a run
    trained on many structures scoreable on one.
    """

    key: str
    vocabulary: tuple[str, ...]
    multi: bool = False
    # Probability a training case is given this field's null in place of its own value.
    dropout_p: float = 0.0
    eval_values: tuple[str, ...] = ()

    @property
    def slots(self) -> int:
        """Columns this field occupies in a prompt row."""
        return len(self.vocabulary) if self.multi else 1


@dataclass(frozen=True)
class PromptConfig:
    """The prompt naming what to segment, and where its embedding meets the network.

    Each field owns a contiguous block of the embedding table whose first row is that field's own
    null, so the fields are asked and answered independently: a location can be dropped while the
    anatomy is still named. One field at offset 0 is the whole table, which is what keeps a
    single-field run identical to one written before fields existed.
    """

    fields: tuple[PromptField, ...]
    width: int = 256
    # `encoder` gates the four feature maps, `decoder` the head's fused features, `both` all five.
    sites: str = "decoder"

    @property
    def offsets(self) -> tuple[int, ...]:
        """The first embedding row of each field's block, which is that field's null."""
        offsets, row = [], 0
        for field in self.fields:
            offsets.append(row)
            row += 1 + len(field.vocabulary)
        return tuple(offsets)

    @property
    def rows(self) -> int:
        return sum(1 + len(field.vocabulary) for field in self.fields)

    @property
    def slots(self) -> int:
        return sum(field.slots for field in self.fields)

    def row(self, values: dict) -> tuple[int, ...]:
        """A prompt row: each field's answers as embedding rows, padded out with its null."""
        row = []
        for field, offset in zip(self.fields, self.offsets):
            indices = [offset + 1 + field.vocabulary.index(name)
                       for name in values.get(field.key, ()) if name in field.vocabulary]
            row.extend(indices + [offset] * (field.slots - len(indices)))
        return tuple(row)

    @property
    def gates_encoder(self) -> bool:
        return self.sites in {"encoder", "both"}

    @property
    def gates_decoder(self) -> bool:
        return self.sites in {"decoder", "both"}


def _prompt_field(declared: dict) -> PromptField:
    key = str(declared.get("key", "location"))
    vocabulary = tuple(str(x) for x in declared.get("vocabulary", ()))
    if not vocabulary:
        raise ValueError(f"data.prompt field {key} needs a vocabulary")
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError(f"data.prompt field {key} repeats a vocabulary entry")
    dropout_p = float(declared.get("dropout_p", 0.0))
    if not 0.0 <= dropout_p < 1.0:
        # Dropout reaches the training subset alone, so 1.0 would train the null embedding only and
        # still hand validation and test the ones it never updated. A run meant to carry no prompt
        # says so with a vocabulary every case maps to the same entry of.
        raise ValueError(f"data.prompt field {key} dropout_p must be at least 0 and below 1")
    eval_values = tuple(str(x) for x in declared.get("eval", ()))
    outside = [name for name in eval_values if name not in vocabulary]
    if outside:
        raise ValueError(f"data.prompt field {key} eval names {outside}, outside its vocabulary")
    return PromptField(
        key=key,
        vocabulary=vocabulary,
        multi=bool(declared.get("multi", False)),
        dropout_p=dropout_p,
        eval_values=eval_values,
    )


@dataclass(frozen=True)
class ExperimentConfig:
    raw_data_dir: Path
    results_dir: Path
    train_dataset: str
    test_datasets: tuple[str, ...]
    test_split: str
    test_splits: dict[str, str]
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
    skeleton_recall_weight: float
    skeleton_tube: bool
    distance_weight_tau: float
    distance_weight_floor: float
    seed: int
    device: str
    patching: "PatchConfig | None"
    augment: "AugmentConfig | None"
    prompt: "PromptConfig | None"
    predict_labels: tuple[str, ...]
    stains: tuple[str, ...]
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
        augment = data.get("augment") or {}
        prompt = data.get("prompt") or {}
        channel_dropout_p = float(data.get("channel_dropout_p", 0.5))
        if not 0.0 <= channel_dropout_p <= 1.0:
            raise ValueError("data.channel_dropout_p must be between 0 and 1")
        prompt_fields = ()
        if prompt:
            if prompt.get("fields") and prompt.get("vocabulary"):
                raise ValueError("data.prompt takes either fields or a single key and vocabulary")
            # A prompt asking one question writes that question at the top level.
            declared = prompt.get("fields") or [
                {"key": prompt.get("key", "location"),
                 "vocabulary": prompt.get("vocabulary", ()),
                 "dropout_p": prompt.get("dropout_p", 0.2)}
            ]
            if prompt.get("sites", "decoder") not in {"encoder", "decoder", "both"}:
                raise ValueError("data.prompt.sites must be encoder, decoder or both")
            prompt_fields = tuple(_prompt_field(field) for field in declared)
            keys = [field.key for field in prompt_fields]
            if len(set(keys)) != len(keys):
                raise ValueError("data.prompt asks the same key twice")
        # Datasets may be given by number alone. Naming them the way the raw data directory does, once
        # and here, is what lets every path built from a config -- the run directory, the caches, the
        # prediction and metrics directories -- read the same as the data it was made from.
        raw_data_dir = Path(data["raw_data_dir"])
        return cls(
            raw_data_dir=raw_data_dir,
            results_dir=Path(data.get("results_dir", "models")),
            train_dataset=resolve(raw_data_dir, data["train_dataset"]),
            test_datasets=tuple(resolve(raw_data_dir, x) for x in data.get("test_datasets", [])),
            test_split=str(data.get("test_split", "Tr")),
            # `test_split` is the default for every evaluation set; a set needing a different
            # one names its own here.
            test_splits={resolve(raw_data_dir, name): str(split)
                         for name, split in (data.get("test_splits") or {}).items()},
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
            skeleton_recall_weight=float(training.get("skeleton_recall_weight", 0.0)),
            skeleton_tube=bool(training.get("skeleton_tube", True)),
            # Weights the cross entropy by distance to the nearest annotated pixel, over a band
            # `distance_weight_tau` pixels wide; 0 weights every pixel alike.
            distance_weight_tau=float(training.get("distance_weight_tau", 0.0)),
            distance_weight_floor=float(training.get("distance_weight_floor", 0.1)),
            seed=int(training.get("seed", 0)),
            device=str(training.get("device", "cuda")),
            # The stains the training images actually carry, where that is narrower than what the
            # dataset declares; empty means take the declaration at its word.
            stains=tuple(str(x).upper() for x in data.get("stains", [])),
            channel_dropout=tuple(str(x).upper() for x in data.get("channel_dropout", [])),
            channel_dropout_p=channel_dropout_p,
            balance_sources=bool(data.get("balance_sources", False)),
            # Absent means no augmentation, so every config written before this existed trains
            # exactly as it did.
            augment=(
                AugmentConfig(
                    hflip=bool(augment.get("hflip", True)),
                    vflip=bool(augment.get("vflip", True)),
                    rotation=float(augment.get("rotation", 0.0)),
                    flip_p=float(augment.get("flip_p", 0.5)),
                    zoom_min=float(augment.get("zoom_min", 1.0)),
                    zoom_max=float(augment.get("zoom_max", 1.0)),
                )
                if augment
                else None
            ),
            # Absent means the network takes no prompt at all, so every config written before this
            # existed builds exactly the network it always did.
            prompt=(
                PromptConfig(
                    fields=prompt_fields,
                    width=int(prompt.get("width", 256)),
                    sites=str(prompt.get("sites", "decoder")),
                )
                if prompt
                else None
            ),
            # The class names the written prediction is mapped onto, in order, so a model trained on
            # many structures can be scored against a dataset labelled with one. Empty writes the
            # model's own class indices.
            predict_labels=tuple(str(x) for x in data.get("predict_labels", [])),
            patching=(
                PatchConfig(
                    patch_size=int(patching.get("patch_size", 1008)),
                    patches_per_case=int(patching.get("patches_per_case", 16)),
                    min_roi_fraction=float(patching.get("min_roi_fraction", 0.05)),
                    roi_threshold=int(patching.get("roi_threshold", 0)),
                    overlap=float(patching.get("overlap", 0.5)),
                    ignore_masked_out=bool(patching.get("ignore_masked_out", False)),
                    respect_channels=bool(patching.get("respect_channels", False)),
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


# What a figure needs to know about a run, for runs that ship no config of their own. nnU-Net and the
# other external trainers write `<model>/<train dataset>/<configuration>/fold_<n>` the same way this
# project does, but keep their own plans files instead of a config.yaml, so everything here is read
# off the directory and the raw data directory the caller names.
PLAIN_RUN_FIELDS = ("raw_data_dir", "train_dataset", "test_split", "test_splits", "patching",
                    "stains", "prompt", "predict_labels")


def describe_run_dir(fold_dir, raw_data_dir=None):
    """The run at `fold_dir`, from its `config.yaml` where it has one.

    Without a config, the training set is the directory it sits under and `raw_data_dir` says where
    the datasets live. `test_split` is left empty: the caller works the split out from the cases that
    were predicted, which is the only thing that answers it when nothing was written down.
    """
    config_path = Path(fold_dir) / "config.yaml"
    if config_path.is_file():
        return ExperimentConfig.from_yaml(config_path)
    if raw_data_dir is None:
        return None
    return SimpleNamespace(
        raw_data_dir=Path(raw_data_dir),
        train_dataset=Path(fold_dir).parents[1].name,
        test_split="",
        test_splits={},
        patching=None,
        stains=(),
        prompt=None,
        predict_labels=(),
    )
