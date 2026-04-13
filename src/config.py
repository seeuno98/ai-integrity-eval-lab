"""Structured configuration dataclasses loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 128


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    logging_steps: int = 50
    seed: int = 42
    fp16: bool = False
    dataloader_num_workers: int = 0


@dataclass
class DataConfig:
    dataset_name: str = "lmsys/toxic-chat"
    dataset_config: str = "toxicchat0124"
    text_field: str = "user_prompt"
    label_field: str = "toxicity"
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


@dataclass
class ArtifactsConfig:
    model_dir: str = "outputs/model"
    metrics_dir: str = "outputs/metrics"
    figures_dir: str = "outputs/figures"
    misclassified_path: str = "outputs/metrics/misclassified.csv"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)


def load_config(path: str | Path = "configs/base.yaml") -> AppConfig:
    """Load configuration from a YAML file and return a typed AppConfig.

    Args:
        path: Path to the YAML config file.

    Returns:
        Populated AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        raw = yaml.safe_load(f)

    model_cfg = ModelConfig(**raw.get("model", {}))
    training_cfg = TrainingConfig(**raw.get("training", {}))
    data_cfg = DataConfig(**raw.get("data", {}))
    artifacts_cfg = ArtifactsConfig(**raw.get("artifacts", {}))

    return AppConfig(
        model=model_cfg,
        training=training_cfg,
        data=data_cfg,
        artifacts=artifacts_cfg,
    )
