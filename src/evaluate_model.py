"""Evaluation: metrics, confusion matrix, PR/ROC curves, misclassified examples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from src.config import AppConfig, load_config
from src.data import load_toxic_chat, subsample
from src.preprocess import build_tokenizer, log_label_distribution, tokenize_dataset
from src.train import compute_metrics, _softmax
from src.utils import ensure_dirs, format_metrics, get_logger, set_seed

logger = get_logger(__name__)


def _compute_confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, int]:
    """Compute TP, FP, TN, FN from flat label/prediction arrays.

    Args:
        labels: Ground-truth binary labels (0 or 1).
        predictions: Predicted binary labels (0 or 1).

    Returns:
        Dictionary with keys tp, fp, tn, fn.
    """
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _save_confusion_matrix_plot(
    cm_dict: dict[str, int], output_path: Path
) -> None:
    """Render and save a confusion matrix heatmap.

    Args:
        cm_dict: Dict with tp, fp, tn, fn.
        output_path: File path for the saved PNG.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        matrix = np.array(
            [[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]]
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Predicted Non-toxic", "Predicted Toxic"],
            yticklabels=["Actual Non-toxic", "Actual Toxic"],
            title="Confusion Matrix",
        )
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(matrix[i, j]),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                    fontsize=14,
                )
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", output_path)
    except Exception as exc:
        logger.warning("Could not save confusion matrix plot: %s", exc)


def _save_roc_curve(
    labels: np.ndarray, probs: np.ndarray, output_path: Path
) -> None:
    """Plot and save an ROC curve.

    Args:
        labels: Ground-truth binary labels.
        probs: Predicted probability for the positive (toxic) class.
        output_path: File path for the saved PNG.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, roc_auc_score

        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("ROC curve saved to %s", output_path)
    except Exception as exc:
        logger.warning("Could not save ROC curve: %s", exc)


def _save_pr_curve(
    labels: np.ndarray, probs: np.ndarray, output_path: Path
) -> None:
    """Plot and save a Precision-Recall curve.

    Args:
        labels: Ground-truth binary labels.
        probs: Predicted probability for the positive (toxic) class.
        output_path: File path for the saved PNG.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score

        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision, label=f"AP = {ap:.3f}")
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("PR curve saved to %s", output_path)
    except Exception as exc:
        logger.warning("Could not save PR curve: %s", exc)


def _save_misclassified(
    raw_texts: list[str],
    labels: np.ndarray,
    predictions: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
    n_top: int = 50,
) -> None:
    """Save the most-confident misclassifications to a CSV file.

    Args:
        raw_texts: Original (un-tokenized) text strings.
        labels: Ground-truth binary labels.
        predictions: Predicted binary labels.
        probs: Predicted probabilities for the positive class.
        output_path: Destination CSV path.
        n_top: Number of examples to save per error type.
    """
    rows = []
    for i, (text, lbl, pred, prob) in enumerate(
        zip(raw_texts, labels, predictions, probs)
    ):
        if lbl != pred:
            rows.append(
                {
                    "index": i,
                    "text": text,
                    "true_label": int(lbl),
                    "predicted_label": int(pred),
                    "toxic_prob": float(prob),
                    "error_type": "FP" if pred == 1 else "FN",
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        # Keep top-N FP and FN sorted by confidence (most wrong first)
        fp = df[df.error_type == "FP"].nlargest(n_top, "toxic_prob")
        fn = df[df.error_type == "FN"].nsmallest(n_top, "toxic_prob")
        df_out = pd.concat([fp, fn]).reset_index(drop=True)
    else:
        df_out = df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    logger.info(
        "Saved %d misclassified examples to %s", len(df_out), output_path
    )


def run_evaluation(config_path: str = "configs/base.yaml", split: str = "test") -> None:
    """Load a fine-tuned model and run comprehensive evaluation.

    Generates metrics JSON, confusion matrix, ROC curve, PR curve, and a
    CSV of the most-confident misclassifications.

    Args:
        config_path: Path to the YAML configuration file.
        split: Dataset split to evaluate on (``"validation"`` or ``"test"``).
    """
    cfg: AppConfig = load_config(config_path)
    set_seed(cfg.training.seed)

    ensure_dirs(cfg.artifacts.metrics_dir, cfg.artifacts.figures_dir)

    model_dir = cfg.artifacts.model_dir
    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"Model directory '{model_dir}' not found. Run training first."
        )

    logger.info("Loading model from '%s' ...", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # ── Prepare data ──────────────────────────────────────────────────────────
    raw = load_toxic_chat(cfg.data)
    raw = subsample(
        raw,
        max_train=cfg.data.max_train_samples,
        max_eval=cfg.data.max_eval_samples,
        seed=cfg.training.seed,
    )

    raw_texts = raw[split][cfg.data.text_field]
    tokenized = tokenize_dataset(raw, tokenizer, cfg.data, cfg.model)
    log_label_distribution(tokenized)

    eval_ds = tokenized[split]

    # ── Inference ─────────────────────────────────────────────────────────────
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        seed=cfg.training.seed,
        fp16=cfg.training.fp16,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Running predictions on '%s' split ...", split)
    pred_output = trainer.predict(eval_ds)
    logits = pred_output.predictions
    labels = pred_output.label_ids.astype(int)
    predictions = np.argmax(logits, axis=-1)
    probs = _softmax(logits)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = {k: float(v) for k, v in pred_output.metrics.items()}
    cm = _compute_confusion_matrix(labels, predictions)
    metrics["confusion_matrix"] = cm
    logger.info("Evaluation metrics:\n%s", format_metrics(metrics, prefix="  "))

    metrics_path = Path(cfg.artifacts.metrics_dir) / f"eval_{split}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    figures_dir = Path(cfg.artifacts.figures_dir)
    _save_confusion_matrix_plot(cm, figures_dir / f"confusion_matrix_{split}.png")
    _save_roc_curve(labels, probs, figures_dir / f"roc_curve_{split}.png")
    _save_pr_curve(labels, probs, figures_dir / f"pr_curve_{split}.png")

    # ── Misclassified ─────────────────────────────────────────────────────────
    _save_misclassified(
        raw_texts=list(raw_texts),
        labels=labels,
        predictions=predictions,
        probs=probs,
        output_path=Path(cfg.artifacts.misclassified_path),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate toxicity classifier")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--split",
        default="test",
        choices=["validation", "test"],
        help="Dataset split to evaluate on",
    )
    args = parser.parse_args()
    run_evaluation(args.config, args.split)
