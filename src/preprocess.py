"""Preprocessing: cleaning, label mapping, and tokenization."""

from __future__ import annotations

from typing import Any

from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import DataConfig, ModelConfig
from src.utils import get_logger

logger = get_logger(__name__)

# Mapping from raw label values to binary integers.
# lmsys/toxic-chat uses 0 (non-toxic) and 1 (toxic) already,
# but we normalise to be safe.
LABEL_MAP: dict[Any, int] = {0: 0, 1: 1, "0": 0, "1": 1, False: 0, True: 1}
ID2LABEL: dict[int, str] = {0: "non-toxic", 1: "toxic"}
LABEL2ID: dict[str, int] = {"non-toxic": 0, "toxic": 1}


def _clean_split(dataset_dict: DatasetDict, text_field: str, label_field: str) -> DatasetDict:
    """Remove rows where text or label is null/empty.

    Args:
        dataset_dict: DatasetDict to clean.
        text_field: Column name for the input text.
        label_field: Column name for the label.

    Returns:
        Cleaned DatasetDict.
    """

    # Validate schema once using train split (or any split)
    sample_split = next(iter(dataset_dict.values()))

    if text_field not in sample_split.column_names:
        raise ValueError(
            f"Text field '{text_field}' not found. Available columns: {sample_split.column_names}"
        )

    if label_field not in sample_split.column_names:
        raise ValueError(
            f"Label field '{label_field}' not found. Available columns: {sample_split.column_names}"
        )

    cleaned: dict[str, Any] = {}
    for split, ds in dataset_dict.items():
        before = len(ds)
        ds = ds.filter(
            lambda row: (
                row[text_field] is not None
                and str(row[text_field]).strip() != ""
                and row[label_field] is not None
            )
        )
        after = len(ds)
        if before != after:
            logger.warning(
                "Dropped %d null/empty rows from '%s' split.", before - after, split
            )
        cleaned[split] = ds
    return DatasetDict(cleaned)


def _map_labels(dataset_dict: DatasetDict, label_field: str) -> DatasetDict:
    """Normalise raw labels to binary integers and rename column to 'labels'.

    The Trainer expects the integer label column to be named ``labels``.

    Args:
        dataset_dict: DatasetDict with a raw label column.
        label_field: Source label column name.

    Returns:
        DatasetDict with a new ``labels`` column (int) and original removed.
    """
    def _remap(batch: dict[str, list]) -> dict[str, list]:
        mapped = []
        for v in batch[label_field]:
            if v not in LABEL_MAP:
                raise ValueError(
                    f"Unexpected label value '{v}'. "
                    f"Expected one of {list(LABEL_MAP.keys())}."
                )
            mapped.append(LABEL_MAP[v])
        return {"labels": mapped}

    result: dict[str, Any] = {}
    for split, ds in dataset_dict.items():
        ds = ds.map(_remap, batched=True, desc=f"Mapping labels [{split}]")
        if label_field != "labels":
            ds = ds.remove_columns([label_field])
        result[split] = ds
    return DatasetDict(result)


def build_tokenizer(model_cfg: ModelConfig) -> PreTrainedTokenizerBase:
    """Instantiate and return the tokenizer for the configured model.

    Args:
        model_cfg: Model configuration containing the HF model name.

    Returns:
        Loaded tokenizer.
    """
    logger.info("Loading tokenizer: %s", model_cfg.name)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name)
    return tokenizer


def tokenize_dataset(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
) -> DatasetDict:
    """Tokenize text fields and set dataset format for PyTorch.

    Steps performed:
    1. Clean null rows.
    2. Map labels to binary integers under the ``labels`` column.
    3. Tokenize with padding and truncation.
    4. Remove raw text and any unused columns.
    5. Set format to ``torch``.

    Args:
        dataset_dict: Raw DatasetDict loaded by :mod:`src.data`.
        tokenizer: Pre-loaded tokenizer.
        data_cfg: Data configuration (field names, etc.).
        model_cfg: Model configuration (max_length).

    Returns:
        Fully preprocessed DatasetDict ready for the Trainer.
    """
    dataset_dict = _clean_split(dataset_dict, data_cfg.text_field, data_cfg.label_field)
    dataset_dict = _map_labels(dataset_dict, data_cfg.label_field)

    def _tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch[data_cfg.text_field],
            truncation=True,
            padding="max_length",
            max_length=model_cfg.max_length,
        )

    tokenized: dict[str, Any] = {}
    for split, ds in dataset_dict.items():
        ds = ds.map(_tokenize, batched=True, desc=f"Tokenizing [{split}]")
        # Keep only the columns the Trainer needs
        keep_cols = {"input_ids", "attention_mask", "labels"}
        if "token_type_ids" in ds.column_names:
            keep_cols.add("token_type_ids")
        remove_cols = [c for c in ds.column_names if c not in keep_cols]
        ds = ds.remove_columns(remove_cols)
        ds.set_format("torch")
        tokenized[split] = ds
        logger.info(
            "Split '%s': %d examples, columns: %s", split, len(ds), ds.column_names
        )

    return DatasetDict(tokenized)


def log_label_distribution(dataset_dict: DatasetDict) -> None:
    """Log per-split label distribution to help detect class imbalance.

    Args:
        dataset_dict: DatasetDict with a ``labels`` column.
    """
    for split, ds in dataset_dict.items():
        labels = ds["labels"]
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        n_toxic = sum(int(l) for l in labels)
        n_total = len(labels)
        pct = 100 * n_toxic / n_total if n_total else 0
        logger.info(
            "[%s] toxic: %d / %d (%.1f%%)", split, n_toxic, n_total, pct
        )
