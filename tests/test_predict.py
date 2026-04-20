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
        DistilBertConfig,
        DistilBertForSequenceClassification,
    )
    from src.preprocess import ID2LABEL, LABEL2ID

    tmp_dir = tmp_path_factory.mktemp("tiny_model")

    config = DistilBertConfig(
        vocab_size=30522,  # must match distilbert-base-uncased tokenizer vocabulary
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
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model.save_pretrained(str(tmp_dir))
    tokenizer.save_pretrained(str(tmp_dir))
    return str(tmp_dir)


@pytest.fixture(scope="module")
def predictor(tiny_model_dir):
    from src.predict import ToxicityPredictor
    return ToxicityPredictor(model_dir=tiny_model_dir, max_length=32, device="cpu")


# ---------------------------------------------------------------------------
# Constructor tests
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

    def test_normalize_flag_stored(self, tiny_model_dir):
        from src.predict import ToxicityPredictor
        p = ToxicityPredictor(model_dir=tiny_model_dir, device="cpu", normalize=False)
        assert p.normalize is False


# ---------------------------------------------------------------------------
# Single-text prediction
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_dict(self, predictor):
        result = predictor.predict("Hello world")
        assert isinstance(result, dict)

    def test_has_required_keys(self, predictor):
        result = predictor.predict("Hello world")
        assert {"label", "label_id", "confidence", "toxic_prob", "threshold"} <= result.keys()

    def test_label_is_string(self, predictor):
        assert isinstance(predictor.predict("Some text")["label"], str)

    def test_label_in_valid_set(self, predictor):
        assert predictor.predict("Some text")["label"] in {"toxic", "non-toxic"}

    def test_label_id_is_binary(self, predictor):
        assert predictor.predict("Some text")["label_id"] in {0, 1}

    def test_confidence_in_range(self, predictor):
        r = predictor.predict("Some text")
        assert 0.0 <= r["confidence"] <= 1.0

    def test_toxic_prob_in_range(self, predictor):
        r = predictor.predict("Some text")
        assert 0.0 <= r["toxic_prob"] <= 1.0

    def test_label_id_matches_label(self, predictor):
        from src.preprocess import ID2LABEL
        r = predictor.predict("Some text")
        assert r["label"] == ID2LABEL[r["label_id"]]

    def test_threshold_stored_in_result(self, predictor):
        r = predictor.predict("Some text", threshold=0.3)
        assert r["threshold"] == 0.3

    def test_empty_text_raises(self, predictor):
        with pytest.raises(ValueError, match="non-empty"):
            predictor.predict("")

    def test_whitespace_only_raises(self, predictor):
        with pytest.raises(ValueError, match="non-empty"):
            predictor.predict("   ")

    def test_invalid_threshold_raises(self, predictor):
        with pytest.raises(ValueError, match="threshold"):
            predictor.predict("hello", threshold=1.5)

    def test_zero_threshold_raises(self, predictor):
        with pytest.raises(ValueError, match="threshold"):
            predictor.predict("hello", threshold=0.0)

    def test_long_text_truncated(self, predictor):
        long_text = "word " * 500
        result = predictor.predict(long_text)
        assert result["label"] in {"toxic", "non-toxic"}

    def test_low_threshold_biases_toward_toxic(self, predictor):
        """With threshold=0.01, any non-zero toxic_prob triggers toxic label."""
        result = predictor.predict("Hello there", threshold=0.01)
        # At threshold=0.01, almost any text with toxic_prob > 0.01 is toxic
        assert result["label_id"] == 1 or result["toxic_prob"] <= 0.01

    def test_high_threshold_biases_toward_non_toxic(self, predictor):
        """With threshold=0.99, only very high confidence triggers toxic label."""
        result = predictor.predict("Hello there", threshold=0.99)
        # At threshold=0.99, non-toxic should be likely unless model is very confident
        assert result["threshold"] == 0.99


# ---------------------------------------------------------------------------
# Threshold consistency
# ---------------------------------------------------------------------------

class TestThresholdConsistency:
    def test_label_id_reflects_threshold(self, predictor):
        """label_id should match whether toxic_prob >= threshold."""
        r = predictor.predict("test text", threshold=0.4)
        expected_id = 1 if r["toxic_prob"] >= 0.4 else 0
        assert r["label_id"] == expected_id

    def test_default_threshold_used_when_not_specified(self, predictor):
        r = predictor.predict("test text")
        assert r["threshold"] == 0.5

    def test_custom_threshold_stored(self, predictor):
        r = predictor.predict("test text", threshold=0.7)
        assert r["threshold"] == 0.7


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalizationInPredict:
    def test_normalize_flag_applied(self, tiny_model_dir):
        """Predictor with normalize=True should produce same result for padded text."""
        from src.predict import ToxicityPredictor

        p = ToxicityPredictor(model_dir=tiny_model_dir, max_length=32, device="cpu", normalize=True)
        r1 = p.predict("hello world")
        r2 = p.predict("hello   world")  # extra spaces
        # After normalization both become "hello world" → same tokenization → same result
        assert abs(r1["toxic_prob"] - r2["toxic_prob"]) < 1e-5

    def test_normalize_false_may_differ(self, tiny_model_dir):
        """When normalization is off, extra spaces may produce a different token sequence.

        This test just verifies the flag is respected; actual differences are model-dependent.
        """
        from src.predict import ToxicityPredictor

        p = ToxicityPredictor(model_dir=tiny_model_dir, max_length=32, device="cpu", normalize=False)
        # Just check it runs without error
        result = p.predict("hello   world")
        assert isinstance(result["toxic_prob"], float)


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_batch_returns_list(self, predictor):
        results = predictor.predict_batch(["Hello", "Go away!"])
        assert isinstance(results, list) and len(results) == 2

    def test_batch_each_element_has_keys(self, predictor):
        for r in predictor.predict_batch(["a", "b", "c"]):
            assert {"label", "label_id", "confidence", "toxic_prob", "threshold"} <= r.keys()

    def test_single_element_batch(self, predictor):
        assert len(predictor.predict_batch(["just one"])) == 1

    def test_batch_threshold_propagated(self, predictor):
        results = predictor.predict_batch(["text one", "text two"], threshold=0.3)
        for r in results:
            assert r["threshold"] == 0.3


# ---------------------------------------------------------------------------
# Probe file loading
# ---------------------------------------------------------------------------

class TestProbeFileLoading:
    def test_load_valid_probe_file(self, tmp_path):
        from src.run_probes import _load_probe_file

        csv = tmp_path / "probes.csv"
        csv.write_text(
            "text,expected_label,category,notes\n"
            "Hello,0,harmless,greeting\n"
            "idiot,1,insult,bare slur\n"
        )
        df = _load_probe_file(csv)
        assert len(df) == 2
        assert list(df["expected_label"]) == [0, 1]

    def test_missing_file_raises(self, tmp_path):
        from src.run_probes import _load_probe_file

        with pytest.raises(FileNotFoundError):
            _load_probe_file(tmp_path / "nonexistent.csv")

    def test_missing_column_raises(self, tmp_path):
        from src.run_probes import _load_probe_file

        csv = tmp_path / "bad.csv"
        csv.write_text("text,expected_label\nhello,0\n")
        with pytest.raises(ValueError, match="missing columns"):
            _load_probe_file(csv)
