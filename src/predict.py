"""CLI and programmatic interface for single-text toxicity prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import load_config
from src.preprocess import ID2LABEL
from src.utils import get_logger

logger = get_logger(__name__)


class ToxicityPredictor:
    """Load a fine-tuned toxicity classifier and run inference.

    Args:
        model_dir: Path to the saved model/tokenizer directory.
        max_length: Maximum tokenization length.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
                Defaults to auto-detection.
    """

    def __init__(
        self,
        model_dir: str,
        max_length: int = 128,
        device: str | None = None,
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

        logger.info("Loading tokenizer and model from '%s' ...", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded on device: %s", self.device)

    def predict(self, text: str) -> dict[str, Any]:
        """Predict toxicity for a single text string.

        Args:
            text: Input text to classify.

        Returns:
            Dictionary with:
            - ``label`` (str): ``"toxic"`` or ``"non-toxic"``
            - ``label_id`` (int): 1 for toxic, 0 for non-toxic
            - ``confidence`` (float): Probability of the predicted label
            - ``toxic_prob`` (float): Probability of the ``"toxic"`` class
        """
        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

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
        label_id = int(probs.argmax().item())
        confidence = float(probs[label_id].item())
        toxic_prob = float(probs[1].item())

        return {
            "label": ID2LABEL[label_id],
            "label_id": label_id,
            "confidence": confidence,
            "toxic_prob": toxic_prob,
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict toxicity for a list of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of prediction dicts (same schema as :meth:`predict`).
        """
        return [self.predict(t) for t in texts]


def main() -> None:
    """CLI entry point for interactive or batch prediction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict toxicity for a given text using a fine-tuned model."
    )
    parser.add_argument("text", nargs="?", help="Text to classify (positional)")
    parser.add_argument("--model-dir", default=None, help="Path to saved model directory")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML config path")
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = args.model_dir or cfg.artifacts.model_dir
    max_length = args.max_length or cfg.model.max_length

    predictor = ToxicityPredictor(model_dir=model_dir, max_length=max_length)

    if args.text:
        result = predictor.predict(args.text)
        print(f"\nText    : {args.text}")
        print(f"Label   : {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Toxic prob: {result['toxic_prob']:.4f}")
    else:
        print("No text provided. Enter text to classify (Ctrl+C to quit):\n")
        while True:
            try:
                text = input("> ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            if not text.strip():
                continue
            result = predictor.predict(text)
            print(
                f"  → {result['label']} "
                f"(confidence={result['confidence']:.3f}, "
                f"toxic_prob={result['toxic_prob']:.3f})\n"
            )


if __name__ == "__main__":
    main()
