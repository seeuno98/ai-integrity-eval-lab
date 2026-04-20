"""Evaluation: metrics, confusion matrix, PR/ROC curves, threshold tuning, and error slices."""

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


# ── Confusion matrix ───────────────────────────────────────────────────────────

def _compute_confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, int]:
    """Compute TP, FP, TN, FN from flat label/prediction arrays."""
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _save_confusion_matrix_plot(cm_dict: dict[str, int], output_path: Path) -> None:
    """Render and save a confusion matrix heatmap."""
    try:
        import matplotlib.pyplot as plt

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


# ── ROC / PR curves ────────────────────────────────────────────────────────────

def _save_roc_curve(labels: np.ndarray, probs: np.ndarray, output_path: Path) -> None:
    """Plot and save an ROC curve."""
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


def _save_pr_curve(labels: np.ndarray, probs: np.ndarray, output_path: Path) -> None:
    """Plot and save a Precision-Recall curve."""
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


# ── Misclassified examples ─────────────────────────────────────────────────────

def _save_misclassified(
    raw_texts: list[str],
    labels: np.ndarray,
    predictions: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
    n_top: int = 50,
) -> None:
    """Save the most-confident misclassifications to a CSV file."""
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
        fp = df[df.error_type == "FP"].nlargest(n_top, "toxic_prob")
        fn = df[df.error_type == "FN"].nsmallest(n_top, "toxic_prob")
        df_out = pd.concat([fp, fn]).reset_index(drop=True)
    else:
        df_out = df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    logger.info("Saved %d misclassified examples to %s", len(df_out), output_path)


# ── Threshold tuning ───────────────────────────────────────────────────────────

def _metrics_at_threshold(
    labels: np.ndarray, probs: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Compute classification metrics for a single decision threshold.

    Args:
        labels: Ground-truth binary labels.
        probs: Predicted probabilities for the toxic class.
        threshold: Cutoff; examples with ``prob >= threshold`` are predicted toxic.

    Returns:
        Dict with precision, recall, f1, accuracy, tp, fp, tn, fn.
    """
    preds = (probs >= threshold).astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _run_threshold_analysis(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold_grid: list[float],
    split: str,
    metrics_dir: Path,
    figures_dir: Path,
) -> None:
    """Sweep thresholds, save CSV + JSON summary + tradeoff plot.

    Args:
        labels: Ground-truth binary labels.
        probs: Predicted toxic-class probabilities.
        threshold_grid: List of threshold values to evaluate.
        split: Dataset split name (used in output filenames).
        metrics_dir: Directory for CSV and JSON outputs.
        figures_dir: Directory for PNG output.
    """
    rows = [_metrics_at_threshold(labels, probs, t) for t in threshold_grid]
    df = pd.DataFrame(rows)

    csv_path = metrics_dir / f"threshold_metrics_{split}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Threshold metrics saved to %s", csv_path)

    # Identify operating points of interest
    best_f1_row = df.loc[df.f1.idxmax()]
    high_recall_rows = df[df.recall >= 0.85]
    high_precision_rows = df[df.precision >= 0.90]

    summary: dict[str, Any] = {
        "best_f1": best_f1_row.to_dict(),
        "high_recall_threshold": (
            high_recall_rows.iloc[0].to_dict() if not high_recall_rows.empty else None
        ),
        "high_precision_threshold": (
            high_precision_rows.iloc[-1].to_dict() if not high_precision_rows.empty else None
        ),
    }

    json_path = metrics_dir / f"threshold_summary_{split}.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Threshold summary saved to %s", json_path)

    logger.info(
        "Best F1=%.4f at threshold=%.2f  (precision=%.4f, recall=%.4f)",
        best_f1_row.f1,
        best_f1_row.threshold,
        best_f1_row.precision,
        best_f1_row.recall,
    )

    # Plot: threshold vs precision / recall / f1
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df.threshold, df.precision, marker="o", label="Precision")
        ax.plot(df.threshold, df.recall, marker="s", label="Recall")
        ax.plot(df.threshold, df.f1, marker="^", label="F1", linewidth=2)
        ax.axvline(best_f1_row.threshold, color="gray", linestyle="--", alpha=0.6,
                   label=f"Best F1 threshold={best_f1_row.threshold:.2f}")
        ax.set(
            xlabel="Decision Threshold",
            ylabel="Score",
            title="Threshold vs Precision / Recall / F1",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(loc="center left")
        fig.tight_layout()
        plot_path = figures_dir / f"threshold_tradeoff_{split}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info("Threshold tradeoff plot saved to %s", plot_path)
    except Exception as exc:
        logger.warning("Could not save threshold tradeoff plot: %s", exc)


# ── Error slices ───────────────────────────────────────────────────────────────

def _compute_error_slices(
    raw_texts: list[str],
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """Compute accuracy broken down by simple diagnostic slices.

    Slices:
    - Text length buckets (short / medium / long by character count)
    - Punctuation presence (text contains ``!`` or ``?``)
    - Non-ASCII presence (text contains characters outside ASCII range)

    Args:
        raw_texts: Original un-tokenized text strings.
        labels: Ground-truth binary labels.
        predictions: Predicted binary labels.

    Returns:
        Nested dict of slice name → metrics (accuracy, n, n_correct).
    """
    texts_arr = np.array(raw_texts)
    char_lens = np.array([len(t) for t in raw_texts])

    short_mask = char_lens < 50
    medium_mask = (char_lens >= 50) & (char_lens < 200)
    long_mask = char_lens >= 200

    has_punct_mask = np.array(
        [any(c in t for c in "!?") for t in raw_texts]
    )
    has_nonascii_mask = np.array(
        [any(ord(c) > 127 for c in t) for t in raw_texts]
    )

    def _slice_metrics(mask: np.ndarray) -> dict[str, Any]:
        if mask.sum() == 0:
            return {"n": 0, "n_correct": 0, "accuracy": None}
        n = int(mask.sum())
        n_correct = int((labels[mask] == predictions[mask]).sum())
        return {"n": n, "n_correct": n_correct, "accuracy": round(n_correct / n, 4)}

    slices = {
        "short_text_lt50chars": _slice_metrics(short_mask),
        "medium_text_50_200chars": _slice_metrics(medium_mask),
        "long_text_gt200chars": _slice_metrics(long_mask),
        "contains_exclamation_or_question": _slice_metrics(has_punct_mask),
        "no_exclamation_or_question": _slice_metrics(~has_punct_mask),
        "contains_non_ascii": _slice_metrics(has_nonascii_mask),
        "ascii_only": _slice_metrics(~has_nonascii_mask),
    }
    return slices


# ── Main evaluation entry point ────────────────────────────────────────────────

# Type alias used in _run_threshold_analysis summary dict
from typing import Any


def run_evaluation(config_path: str = "configs/base.yaml", split: str = "test") -> None:
    """Load a fine-tuned model and run comprehensive evaluation.

    Generates metrics JSON, confusion matrix, ROC curve, PR curve, threshold
    analysis, error slices, and a CSV of the most-confident misclassifications.

    Args:
        config_path: Path to the YAML configuration file.
        split: Dataset split to evaluate on (``"validation"`` or ``"test"``).
    """
    cfg: AppConfig = load_config(config_path)
    set_seed(cfg.training.seed)

    metrics_dir = Path(cfg.artifacts.metrics_dir)
    figures_dir = Path(cfg.artifacts.figures_dir)
    ensure_dirs(metrics_dir, figures_dir)

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

    raw_texts = list(raw[split][cfg.data.text_field])
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
    probs = _softmax(logits)[:, 1]

    threshold = cfg.inference.default_threshold
    predictions = (probs >= threshold).astype(int)

    # ── Core metrics ──────────────────────────────────────────────────────────
    metrics = {k: float(v) for k, v in pred_output.metrics.items()}
    cm = _compute_confusion_matrix(labels, predictions)
    metrics["confusion_matrix"] = cm
    metrics["threshold_used"] = threshold
    logger.info("Evaluation metrics:\n%s", format_metrics(metrics, prefix="  "))

    metrics_path = metrics_dir / f"eval_{split}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _save_confusion_matrix_plot(cm, figures_dir / f"confusion_matrix_{split}.png")
    _save_roc_curve(labels, probs, figures_dir / f"roc_curve_{split}.png")
    _save_pr_curve(labels, probs, figures_dir / f"pr_curve_{split}.png")

    # ── Misclassified ─────────────────────────────────────────────────────────
    _save_misclassified(
        raw_texts=raw_texts,
        labels=labels,
        predictions=predictions,
        probs=probs,
        output_path=Path(cfg.artifacts.misclassified_path),
    )

    # ── Threshold tuning ──────────────────────────────────────────────────────
    logger.info("Running threshold analysis ...")
    _run_threshold_analysis(
        labels=labels,
        probs=probs,
        threshold_grid=cfg.inference.threshold_grid,
        split=split,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
    )

    # ── Error slices ──────────────────────────────────────────────────────────
    logger.info("Computing error slices ...")
    slices = _compute_error_slices(raw_texts, labels, predictions)
    slices_path = metrics_dir / f"error_slices_{split}.json"
    with slices_path.open("w") as f:
        json.dump(slices, f, indent=2)
    logger.info("Error slices saved to %s", slices_path)

    for slice_name, s in slices.items():
        if s["accuracy"] is not None:
            logger.info(
                "  Slice %-40s  n=%d  accuracy=%.4f",
                slice_name,
                s["n"],
                s["accuracy"],
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
