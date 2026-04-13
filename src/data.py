"""Dataset loading utilities for lmsys/toxic-chat."""

from __future__ import annotations

from typing import Optional

from datasets import DatasetDict, load_dataset

from src.config import DataConfig
from src.utils import get_logger

logger = get_logger(__name__)


def load_toxic_chat(cfg: DataConfig) -> DatasetDict:
    """Load the lmsys/toxic-chat dataset from Hugging Face Hub.

    The dataset ships with a single 'train' split.  This function
    carves out validation and test splits deterministically so that
    results are reproducible across runs.

    Args:
        cfg: Data configuration containing dataset name, config string,
             and split ratios.

    Returns:
        DatasetDict with keys ``train``, ``validation``, and ``test``.

    Raises:
        ValueError: If split ratios sum to >= 1.0, leaving no training data.
    """
    if cfg.val_split_ratio + cfg.test_split_ratio >= 1.0:
        raise ValueError(
            "val_split_ratio + test_split_ratio must be < 1.0; "
            f"got {cfg.val_split_ratio + cfg.test_split_ratio}"
        )

    logger.info(
        "Loading dataset '%s' (config=%s) from Hugging Face Hub ...",
        cfg.dataset_name,
        cfg.dataset_config,
    )

    raw: DatasetDict = load_dataset(cfg.dataset_name, cfg.dataset_config)

    # toxic-chat only has a 'train' split; we create val/test from it.
    if "validation" not in raw and "test" not in raw:
        logger.info("Splitting train → train / validation / test ...")
        train_val_test = raw["train"].train_test_split(
            test_size=cfg.test_split_ratio, seed=42
        )
        train_val = train_val_test["train"].train_test_split(
            test_size=cfg.val_split_ratio / (1.0 - cfg.test_split_ratio),
            seed=42,
        )
        raw = DatasetDict(
            {
                "train": train_val["train"],
                "validation": train_val["test"],
                "test": train_val_test["test"],
            }
        )
    elif "validation" not in raw:
        train_val = raw["train"].train_test_split(
            test_size=cfg.val_split_ratio, seed=42
        )
        raw = DatasetDict(
            {
                "train": train_val["train"],
                "validation": train_val["test"],
                "test": raw["test"],
            }
        )

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(raw["train"]),
        len(raw["validation"]),
        len(raw["test"]),
    )
    return raw


def subsample(
    dataset_dict: DatasetDict,
    max_train: Optional[int],
    max_eval: Optional[int],
    seed: int = 42,
) -> DatasetDict:
    """Optionally subsample splits for fast experimentation.

    Args:
        dataset_dict: Full DatasetDict with train/validation/test.
        max_train: Maximum number of training examples.  ``None`` keeps all.
        max_eval: Maximum number of eval/test examples.  ``None`` keeps all.
        seed: Shuffle seed for reproducibility.

    Returns:
        DatasetDict with the (possibly reduced) splits.
    """
    splits = {}
    for split, ds in dataset_dict.items():
        limit = max_train if split == "train" else max_eval
        if limit is not None and len(ds) > limit:
            ds = ds.shuffle(seed=seed).select(range(limit))
            logger.info("Subsampled '%s' to %d examples.", split, limit)
        splits[split] = ds
    return DatasetDict(splits)
