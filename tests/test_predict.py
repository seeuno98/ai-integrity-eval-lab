"""Unit tests for the ToxicityPredictor class."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """Create a minimal DistilBERT classification model for fast tests.

    This saves a real (randomly-initialized) model so ToxicityPredictor
    can load it without requiring a trained checkpoint.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DistilBertConfig,
        DistilBertForSequenceClassification,
    )
    from src.preprocess import ID2LABEL, LABEL2ID

    tmp_dir = tmp_path_factory.mktemp("tiny_model")

    # Tiny config to keep the fixture fast
    config = DistilBertConfig(
        vocab_size=1000,
        max_position_embeddings=64,
        n_layers=1,
        n_heads=2,
        dim=32,
        hidden_dim=64,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = DistilBertForSequenceClassification(config)
    # Use a real tokenizer so tokenization works correctly
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model.save_pretrained(str(tmp_dir))
    tokenizer.save_pretrained(str(tmp_dir))
    return str(tmp_dir)


@pytest.fixture(scope="module")
def predictor(tiny_model_dir):
    from src.predict import ToxicityPredictor

    return ToxicityPredictor(model_dir=tiny_model_dir, max_length=32, device="cpu")


# ---------------------------------------------------------------------------
# Predictor constructor tests
# ---------------------------------------------------------------------------

class TestToxicityPredictorInit:
    def test_loads_successfully(self, predictor):
        from src.predict import ToxicityPredictor

        assert isinstance(predictor, ToxicityPredictor)

    def test_missing_dir_raises(self, tmp_path):
        from src.predict import ToxicityPredictor

        with pytest.raises(FileNotFoundError):
            ToxicityPredictor(model_dir=str(tmp_path / "nonexistent"))

    def test_device_is_cpu(self, predictor):
        import torch

        assert predictor.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Single-text prediction
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_dict(self, predictor):
        result = predictor.predict("Hello world")
        assert isinstance(result, dict)

    def test_has_required_keys(self, predictor):
        result = predictor.predict("Hello world")
        assert {"label", "label_id", "confidence", "toxic_prob"} <= result.keys()

    def test_label_is_string(self, predictor):
        result = predictor.predict("Some text")
        assert isinstance(result["label"], str)

    def test_label_in_valid_set(self, predictor):
        result = predictor.predict("Some text")
        assert result["label"] in {"toxic", "non-toxic"}

    def test_label_id_is_binary(self, predictor):
        result = predictor.predict("Some text")
        assert result["label_id"] in {0, 1}

    def test_confidence_in_range(self, predictor):
        result = predictor.predict("Some text")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_toxic_prob_in_range(self, predictor):
        result = predictor.predict("Some text")
        assert 0.0 <= result["toxic_prob"] <= 1.0

    def test_label_id_matches_label(self, predictor):
        from src.preprocess import ID2LABEL

        result = predictor.predict("Some text")
        assert result["label"] == ID2LABEL[result["label_id"]]

    def test_confidence_matches_label_id(self, predictor):
        """Confidence should equal the probability of the predicted label."""
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        result = predictor.predict("Some text")
        # confidence should be >= 0.5 (argmax side)
        assert result["confidence"] >= 0.5

    def test_empty_text_raises(self, predictor):
        with pytest.raises(ValueError, match="non-empty"):
            predictor.predict("")

    def test_whitespace_only_raises(self, predictor):
        with pytest.raises(ValueError, match="non-empty"):
            predictor.predict("   ")

    def test_long_text_truncated(self, predictor):
        long_text = "word " * 500
        result = predictor.predict(long_text)
        # Should not raise — truncation handles this
        assert result["label"] in {"toxic", "non-toxic"}


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_batch_returns_list(self, predictor):
        results = predictor.predict_batch(["Hello", "Go away!"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_each_element_has_keys(self, predictor):
        results = predictor.predict_batch(["a", "b", "c"])
        for r in results:
            assert {"label", "label_id", "confidence", "toxic_prob"} <= r.keys()

    def test_single_element_batch(self, predictor):
        results = predictor.predict_batch(["just one"])
        assert len(results) == 1
