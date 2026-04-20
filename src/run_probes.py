"""Manual probe runner: classify hand-curated examples and produce a diagnostic report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AppConfig, load_config
from src.predict import ToxicityPredictor
from src.utils import ensure_dirs, get_logger, set_seed

logger = get_logger(__name__)


def _load_probe_file(probe_path: str | Path) -> pd.DataFrame:
    """Load and validate the probe CSV.

    Expected columns: text, expected_label, category, notes.

    Args:
        probe_path: Path to the probe CSV file.

    Returns:
        Validated DataFrame.

    Raises:
        FileNotFoundError: If the probe file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(probe_path)
    if not path.exists():
        raise FileNotFoundError(f"Probe file not found: '{path}'")

    df = pd.read_csv(path)
    required = {"text", "expected_label", "category", "notes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Probe file is missing columns: {missing}")

    df["expected_label"] = df["expected_label"].astype(int)
    df["text"] = df["text"].astype(str)
    return df


def _run_inference(
    df: pd.DataFrame,
    predictor: ToxicityPredictor,
    threshold: float,
) -> pd.DataFrame:
    """Run the predictor on every row and append result columns.

    Args:
        df: Probe DataFrame with a ``text`` column.
        predictor: Loaded ToxicityPredictor.
        threshold: Decision threshold for labelling.

    Returns:
        DataFrame with additional columns: predicted_label, toxic_probability,
        confidence, is_correct.
    """
    predicted_labels = []
    toxic_probs = []
    confidences = []

    for text in df["text"]:
        result = predictor.predict(str(text), threshold=threshold)
        predicted_labels.append(result["label_id"])
        toxic_probs.append(result["toxic_prob"])
        confidences.append(result["confidence"])

    df = df.copy()
    df["predicted_label"] = predicted_labels
    df["toxic_probability"] = toxic_probs
    df["confidence"] = confidences
    df["is_correct"] = df["expected_label"] == df["predicted_label"]
    return df


def _compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute overall and per-category accuracy, precision, recall, F1.

    Args:
        df: Probe results DataFrame with expected_label, predicted_label,
            is_correct, and category columns.

    Returns:
        Summary dict with overall and per_category keys.
    """
    def _metrics_for(subset: pd.DataFrame) -> dict[str, Any]:
        n = len(subset)
        if n == 0:
            return {"n": 0, "accuracy": None}
        n_correct = int(subset.is_correct.sum())
        tp = int(((subset.expected_label == 1) & (subset.predicted_label == 1)).sum())
        fp = int(((subset.expected_label == 0) & (subset.predicted_label == 1)).sum())
        fn = int(((subset.expected_label == 1) & (subset.predicted_label == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)  # type: ignore[operator]
            if precision and recall
            else None
        )
        return {
            "n": n,
            "n_correct": n_correct,
            "accuracy": round(n_correct / n, 4),
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        }

    overall = _metrics_for(df)
    per_category: dict[str, Any] = {}
    for cat, group in df.groupby("category"):
        per_category[str(cat)] = _metrics_for(group)

    return {"overall": overall, "per_category": per_category}


def _build_markdown_report(
    df: pd.DataFrame,
    summary: dict[str, Any],
    threshold: float,
    output_path: Path,
) -> None:
    """Generate a human-readable Markdown diagnostic report.

    Args:
        df: Full probe results DataFrame.
        summary: Summary metrics dict.
        threshold: Decision threshold that was used.
        output_path: File path for the report.
    """
    lines: list[str] = []
    lines.append("# Manual Probe Report\n")
    lines.append(f"Decision threshold: **{threshold:.2f}**\n")

    overall = summary["overall"]
    lines.append("## Overall Results\n")
    lines.append(
        f"- **Total examples**: {overall['n']}\n"
        f"- **Correct**: {overall['n_correct']}\n"
        f"- **Accuracy**: {overall['accuracy']:.4f}\n"
        f"- **Precision**: {overall['precision']}\n"
        f"- **Recall**: {overall['recall']}\n"
        f"- **F1**: {overall['f1']}\n"
    )

    lines.append("## Results by Category\n")
    lines.append("| Category | N | Accuracy | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|")
    for cat, m in sorted(summary["per_category"].items()):
        acc = f"{m['accuracy']:.3f}" if m["accuracy"] is not None else "—"
        prec = f"{m['precision']:.3f}" if m["precision"] is not None else "—"
        rec = f"{m['recall']:.3f}" if m["recall"] is not None else "—"
        f1 = f"{m['f1']:.3f}" if m["f1"] is not None else "—"
        lines.append(f"| {cat} | {m['n']} | {acc} | {prec} | {rec} | {f1} |")
    lines.append("")

    # Failure category: worst accuracy categories
    cat_accs = {
        cat: m["accuracy"]
        for cat, m in summary["per_category"].items()
        if m["accuracy"] is not None
    }
    if cat_accs:
        worst_cat = min(cat_accs, key=lambda c: cat_accs[c])
        lines.append(f"**Biggest failure category**: `{worst_cat}` "
                     f"(accuracy={cat_accs[worst_cat]:.3f})\n")

    errors = df[~df.is_correct]

    # False positives (predicted toxic but not)
    fp = errors[errors.expected_label == 0].nlargest(5, "toxic_probability")
    if not fp.empty:
        lines.append("## Representative False Positives (predicted toxic, actually non-toxic)\n")
        lines.append("| Text | Toxic Prob |")
        lines.append("|---|---|")
        for _, row in fp.iterrows():
            safe_text = str(row.text).replace("|", "\\|")
            lines.append(f"| {safe_text} | {row.toxic_probability:.3f} |")
        lines.append("")

    # False negatives (predicted non-toxic but actually toxic)
    fn = errors[errors.expected_label == 1].nsmallest(5, "toxic_probability")
    if not fn.empty:
        lines.append("## Representative False Negatives (predicted non-toxic, actually toxic)\n")
        lines.append("| Text | Toxic Prob |")
        lines.append("|---|---|")
        for _, row in fn.iterrows():
            safe_text = str(row.text).replace("|", "\\|")
            lines.append(f"| {safe_text} | {row.toxic_probability:.3f} |")
        lines.append("")

    lines.append("## Observations\n")
    lines.append(
        "- Punctuation sensitivity: check whether adding/removing punctuation changes the prediction.\n"
        "- Negation: verify that negated insults are not misclassified as toxic.\n"
        "- Short insults: single-word slurs may fall below the model's confidence threshold.\n"
        "- Multilingual/slang: non-English or censored profanity may be under-represented in training.\n"
    )

    output_path.write_text("\n".join(lines))
    logger.info("Probe report saved to %s", output_path)


def _print_summary(summary: dict[str, Any], threshold: float) -> None:
    """Print a concise summary table to stdout."""
    overall = summary["overall"]
    print("\n" + "=" * 60)
    print(" PROBE RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Threshold : {threshold:.2f}")
    print(f"  Total     : {overall['n']}")
    print(f"  Correct   : {overall['n_correct']}")
    print(f"  Accuracy  : {overall['accuracy']:.4f}")
    print(f"  Precision : {overall['precision']}")
    print(f"  Recall    : {overall['recall']}")
    print(f"  F1        : {overall['f1']}")
    print()
    print("  Per-category accuracy:")
    for cat, m in sorted(summary["per_category"].items()):
        acc_str = f"{m['accuracy']:.3f}" if m["accuracy"] is not None else "—"
        marker = "" if m["accuracy"] is None or m["accuracy"] >= 0.75 else "  ← needs attention"
        print(f"    {cat:<40}  {acc_str}{marker}")
    print("=" * 60 + "\n")


def run_probes(config_path: str = "configs/base.yaml") -> None:
    """Load the saved model and run all manual probe examples.

    Outputs:
    - ``outputs/probes/probe_results.csv``
    - ``outputs/probes/probe_summary.json``
    - ``outputs/probes/probe_report.md``

    Args:
        config_path: Path to the YAML configuration file.
    """
    cfg: AppConfig = load_config(config_path)
    set_seed(cfg.training.seed)

    output_dir = Path(cfg.probes.output_dir)
    ensure_dirs(output_dir)

    # ── Load predictor ────────────────────────────────────────────────────────
    model_dir = cfg.artifacts.model_dir
    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"Model directory '{model_dir}' not found. Run training first."
        )

    threshold = cfg.inference.default_threshold
    predictor = ToxicityPredictor(
        model_dir=model_dir,
        max_length=cfg.model.max_length,
        normalize=cfg.data.normalize_text,
    )

    # ── Load probes ───────────────────────────────────────────────────────────
    df = _load_probe_file(cfg.probes.probe_file)
    logger.info("Loaded %d probe examples from '%s'", len(df), cfg.probes.probe_file)

    # ── Inference ─────────────────────────────────────────────────────────────
    df = _run_inference(df, predictor, threshold)

    # ── Save results CSV ──────────────────────────────────────────────────────
    results_path = output_dir / "probe_results.csv"
    df.to_csv(results_path, index=False)
    logger.info("Probe results saved to %s", results_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = _compute_summary(df)
    summary_path = output_dir / "probe_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Probe summary saved to %s", summary_path)

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = output_dir / "probe_report.md"
    _build_markdown_report(df, summary, threshold, report_path)

    # ── Stdout ────────────────────────────────────────────────────────────────
    _print_summary(summary, threshold)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run manual toxicity probes")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    run_probes(args.config)
