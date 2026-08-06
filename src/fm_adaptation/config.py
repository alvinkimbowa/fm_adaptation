from dataclasses import dataclass
from pathlib import Path

import yaml


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
    probe_name: str
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    seed: int
    device: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        data = cfg["data"]
        model = cfg["model"]
        training = cfg["training"]
        return cls(
            raw_data_dir=Path(data["raw_data_dir"]),
            results_dir=Path(data.get("results_dir", "models")),
            train_dataset=str(data["train_dataset"]),
            test_datasets=tuple(str(x) for x in data.get("test_datasets", [])),
            test_split=str(data.get("test_split", "Tr")),
            fold=str(data["fold"]),
            model_name=str(model["name"]),
            checkpoint=model.get("checkpoint"),
            probe_name=str(model["probe"]),
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            num_workers=int(training.get("num_workers", 4)),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training.get("weight_decay", 0.0)),
            seed=int(training.get("seed", 0)),
            device=str(training.get("device", "cuda")),
        )

    @property
    def run_dir(self) -> Path:
        return (
            self.results_dir
            / self.model_name
            / self.train_dataset
            / self.probe_name
            / f"fold_{self.fold}"
        )

