"""Training entry point: fine-tune DistilBERT on toxic-chat."""

from __future__ import annotations

import json
from pathlib import Path

import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from src.config import AppConfig, load_config
from src.data import load_toxic_chat, subsample
from src.preprocess import (
    ID2LABEL,
    LABEL2ID,
    build_tokenizer,
    log_label_distribution,
    tokenize_dataset,
)
from src.utils import ensure_dirs, format_metrics, get_logger, set_seed

logger = get_logger(__name__)

# Metrics loaded once at module level for efficiency.
_accuracy = evaluate.load("accuracy")
_precision = evaluate.load("precision")
_recall = evaluate.load("recall")
_f1 = evaluate.load("f1")


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute classification metrics for the Trainer callback.

    Calculates accuracy, precision, recall, F1 (binary), and ROC-AUC
    from raw logits and integer labels.

    Args:
        eval_pred: ``EvalPrediction`` namedtuple with ``predictions``
                   (logits array) and ``label_ids`` (int array).

    Returns:
        Dictionary of metric name → float value.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Soft probabilities for ROC-AUC
    probs = _softmax(logits)[:, 1]

    acc = _accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    prec = _precision.compute(
        predictions=predictions, references=labels, average="binary"
    )["precision"]
    rec = _recall.compute(
        predictions=predictions, references=labels, average="binary"
    )["recall"]
    f1 = _f1.compute(
        predictions=predictions, references=labels, average="binary"
    )["f1"]

    metrics: dict[str, float] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    try:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(labels, probs)
        metrics["roc_auc"] = auc
    except Exception:
        pass  # ROC-AUC may fail with single-class batches during warm-up

    return metrics


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def run_training(config_path: str = "configs/base.yaml") -> None:
    """Execute the full training pipeline.

    Loads config, prepares data, fine-tunes the model with the HF Trainer,
    saves model artifacts and final metrics.

    Args:
        config_path: Path to the YAML configuration file.
    """
    cfg: AppConfig = load_config(config_path)
    set_seed(cfg.training.seed)

    ensure_dirs(
        cfg.training.output_dir,
        cfg.artifacts.model_dir,
        cfg.artifacts.metrics_dir,
        cfg.artifacts.figures_dir,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    raw = load_toxic_chat(cfg.data)
    raw = subsample(
        raw,
        max_train=cfg.data.max_train_samples,
        max_eval=cfg.data.max_eval_samples,
        seed=cfg.training.seed,
    )

    tokenizer = build_tokenizer(cfg.model)
    tokenized = tokenize_dataset(raw, tokenizer, cfg.data, cfg.model)
    log_label_distribution(tokenized)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Initialising model: %s", cfg.model.name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=cfg.model.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        eval_strategy=cfg.training.evaluation_strategy,
        save_strategy=cfg.training.save_strategy,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        logging_steps=cfg.training.logging_steps,
        seed=cfg.training.seed,
        fp16=cfg.training.fp16,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training ...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    logger.info("Saving model and tokenizer to '%s' ...", cfg.artifacts.model_dir)
    trainer.save_model(cfg.artifacts.model_dir)
    tokenizer.save_pretrained(cfg.artifacts.model_dir)

    # ── Final eval on validation ──────────────────────────────────────────────
    logger.info("Running final evaluation on validation split ...")
    val_metrics = trainer.evaluate(tokenized["validation"])
    logger.info("Validation metrics:\n%s", format_metrics(val_metrics, prefix="  "))

    metrics_path = Path(cfg.artifacts.metrics_dir) / "train_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(val_metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train toxicity classifier")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    run_training(args.config)
