"""CLI and programmatic interface for single-text toxicity prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import load_config
from src.preprocess import ID2LABEL, normalize_text
from src.utils import get_logger

logger = get_logger(__name__)

_DEFAULT_THRESHOLD = 0.5


class ToxicityPredictor:
    """Load a fine-tuned toxicity classifier and run inference.

    Args:
        model_dir: Path to the saved model/tokenizer directory.
        max_length: Maximum tokenization length.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
                Defaults to auto-detection.
        normalize: Whether to apply text normalization before tokenization.
    """

    def __init__(
        self,
        model_dir: str,
        max_length: int = 128,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        model_dir = str(model_dir)
        if not Path(model_dir).exists():
            raise FileNotFoundError(
                f"Model directory not found: '{model_dir}'. "
                "Run training first."
            )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.max_length = max_length
        self.normalize = normalize

        logger.info("Loading tokenizer and model from '%s' ...", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded on device: %s", self.device)

    def predict(self, text: str, threshold: float = _DEFAULT_THRESHOLD) -> dict[str, Any]:
        """Predict toxicity for a single text string.

        The label is determined by comparing the ``toxic_prob`` to ``threshold``:
        - If ``toxic_prob >= threshold`` → ``"toxic"`` (label_id=1)
        - Otherwise → ``"non-toxic"`` (label_id=0)

        Args:
            text: Input text to classify.
            threshold: Decision threshold for the toxic class.
                       Values below 0.5 increase recall (catch more toxicity);
                       values above 0.5 increase precision.

        Returns:
            Dictionary with:
            - ``label`` (str): ``"toxic"`` or ``"non-toxic"``
            - ``label_id`` (int): 1 for toxic, 0 for non-toxic
            - ``confidence`` (float): Probability of the predicted label
            - ``toxic_prob`` (float): Raw model probability for the ``"toxic"`` class
            - ``threshold`` (float): Decision threshold used

        Raises:
            ValueError: If ``text`` is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1); got {threshold}")

        if self.normalize:
            text = normalize_text(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze()
        toxic_prob = float(probs[1].item())

        label_id = 1 if toxic_prob >= threshold else 0
        confidence = float(probs[label_id].item())

        return {
            "label": ID2LABEL[label_id],
            "label_id": label_id,
            "confidence": confidence,
            "toxic_prob": toxic_prob,
            "threshold": threshold,
        }

    def predict_batch(
        self, texts: list[str], threshold: float = _DEFAULT_THRESHOLD
    ) -> list[dict[str, Any]]:
        """Predict toxicity for a list of texts.

        Args:
            texts: List of input strings.
            threshold: Decision threshold applied to each prediction.

        Returns:
            List of prediction dicts (same schema as :meth:`predict`).
        """
        return [self.predict(t, threshold=threshold) for t in texts]


def main() -> None:
    """CLI entry point for one-shot or interactive prediction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict toxicity for a given text using a fine-tuned model."
    )
    # Accept text either as a named flag or as a positional argument for
    # backwards compatibility.
    parser.add_argument("--text", default=None, help="Text to classify")
    parser.add_argument("positional_text", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", default=None, help="Path to saved model directory")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML config path")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for the toxic class (default: from config)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = args.model_dir or cfg.artifacts.model_dir
    max_length = args.max_length or cfg.model.max_length
    threshold = args.threshold if args.threshold is not None else cfg.inference.default_threshold
    normalize = not args.no_normalize

    predictor = ToxicityPredictor(
        model_dir=model_dir, max_length=max_length, normalize=normalize
    )

    # Resolve text: --text flag takes priority over positional argument
    text = args.text or args.positional_text

    if text:
        result = predictor.predict(text, threshold=threshold)
        print(f"\nText      : {text}")
        print(f"Label     : {result['label']}")
        print(f"Toxic prob: {result['toxic_prob']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Threshold : {result['threshold']:.2f}")
    else:
        print(
            f"Interactive mode (threshold={threshold:.2f}, "
            f"normalize={normalize}). Enter text or Ctrl+C to quit.\n"
        )
        while True:
            try:
                text = input("> ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            if not text.strip():
                continue
            result = predictor.predict(text, threshold=threshold)
            print(
                f"  → {result['label']} "
                f"(toxic_prob={result['toxic_prob']:.3f}, "
                f"confidence={result['confidence']:.3f})\n"
            )


if __name__ == "__main__":
    main()
