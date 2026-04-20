"""Unit tests for preprocessing utilities."""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(rows: list[dict]) -> DatasetDict:
    """Wrap a list of dicts in a DatasetDict with a single 'train' split."""
    return DatasetDict({"train": Dataset.from_list(rows)})


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_strips_leading_trailing_whitespace(self):
        from src.preprocess import normalize_text
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_internal_spaces(self):
        from src.preprocess import normalize_text
        assert normalize_text("hello   world") == "hello world"

    def test_collapses_tabs(self):
        from src.preprocess import normalize_text
        assert normalize_text("hello\tworld") == "hello world"

    def test_collapses_newlines(self):
        from src.preprocess import normalize_text
        assert normalize_text("hello\nworld") == "hello world"

    def test_mixed_whitespace(self):
        from src.preprocess import normalize_text
        assert normalize_text("  hello \t\n world  ") == "hello world"

    def test_unicode_nfkc_fullwidth(self):
        from src.preprocess import normalize_text
        # Full-width ASCII characters should be normalized to ASCII equivalents
        result = normalize_text("\uff48\uff45\uff4c\uff4c\uff4f")  # ｈｅｌｌｏ
        assert result == "hello"

    def test_unicode_nfkc_ligature(self):
        from src.preprocess import normalize_text
        # ﬁ ligature → fi
        result = normalize_text("\ufb01ne")
        assert result == "fine"

    def test_preserves_punctuation(self):
        from src.preprocess import normalize_text
        text = "You are an idiot!"
        assert normalize_text(text) == text

    def test_empty_string_returns_empty(self):
        from src.preprocess import normalize_text
        assert normalize_text("") == ""

    def test_only_whitespace_returns_empty(self):
        from src.preprocess import normalize_text
        assert normalize_text("   \t\n  ") == ""

    def test_already_clean_string_unchanged(self):
        from src.preprocess import normalize_text
        text = "This is a clean sentence."
        assert normalize_text(text) == text


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

class TestLabelMap:
    def test_integer_labels_mapped(self):
        from src.preprocess import LABEL_MAP
        assert LABEL_MAP[0] == 0
        assert LABEL_MAP[1] == 1

    def test_string_labels_mapped(self):
        from src.preprocess import LABEL_MAP
        assert LABEL_MAP["0"] == 0
        assert LABEL_MAP["1"] == 1

    def test_bool_labels_mapped(self):
        from src.preprocess import LABEL_MAP
        assert LABEL_MAP[False] == 0
        assert LABEL_MAP[True] == 1

    def test_id2label_covers_both_classes(self):
        from src.preprocess import ID2LABEL
        assert set(ID2LABEL.keys()) == {0, 1}

    def test_label2id_round_trips(self):
        from src.preprocess import ID2LABEL, LABEL2ID
        for lid, name in ID2LABEL.items():
            assert LABEL2ID[name] == lid


# ---------------------------------------------------------------------------
# Null / empty row cleaning
# ---------------------------------------------------------------------------

class TestCleanSplit:
    def _run(self, rows, text_field="text", label_field="toxicity"):
        from src.preprocess import _clean_split
        dd = _make_dataset(rows)
        return _clean_split(dd, text_field, label_field)

    def test_removes_null_text(self):
        rows = [{"text": None, "toxicity": 0}, {"text": "hello", "toxicity": 0}]
        assert len(self._run(rows)["train"]) == 1

    def test_removes_empty_text(self):
        rows = [{"text": "   ", "toxicity": 0}, {"text": "valid text", "toxicity": 1}]
        assert len(self._run(rows)["train"]) == 1

    def test_removes_null_label(self):
        rows = [{"text": "ok text", "toxicity": None}, {"text": "valid", "toxicity": 1}]
        assert len(self._run(rows)["train"]) == 1

    def test_keeps_valid_rows(self):
        rows = [{"text": "first", "toxicity": 0}, {"text": "second", "toxicity": 1}]
        assert len(self._run(rows)["train"]) == 2

    def test_raises_on_missing_text_field(self):
        from src.preprocess import _clean_split
        dd = _make_dataset([{"wrong": "hi", "toxicity": 0}])
        with pytest.raises(ValueError, match="Text field"):
            _clean_split(dd, "text", "toxicity")

    def test_raises_on_missing_label_field(self):
        from src.preprocess import _clean_split
        dd = _make_dataset([{"text": "hi", "wrong": 0}])
        with pytest.raises(ValueError, match="Label field"):
            _clean_split(dd, "text", "toxicity")


# ---------------------------------------------------------------------------
# Label remapping
# ---------------------------------------------------------------------------

class TestMapLabels:
    def _run(self, rows, label_field="toxicity"):
        from src.preprocess import _map_labels
        dd = _make_dataset(rows)
        return _map_labels(dd, label_field)

    def test_creates_labels_column(self):
        rows = [{"text": "a", "toxicity": 0}, {"text": "b", "toxicity": 1}]
        result = self._run(rows)
        assert "labels" in result["train"].column_names

    def test_integer_values_correct(self):
        rows = [{"text": "a", "toxicity": 0}, {"text": "b", "toxicity": 1}]
        result = self._run(rows)
        assert list(result["train"]["labels"]) == [0, 1]

    def test_original_label_column_removed(self):
        rows = [{"text": "a", "toxicity": 0}]
        assert "toxicity" not in self._run(rows)["train"].column_names

    def test_unexpected_label_raises(self):
        from src.preprocess import _map_labels
        dd = _make_dataset([{"text": "a", "toxicity": 99}])
        with pytest.raises(ValueError, match="Unexpected label"):
            _map_labels(dd, "toxicity")


# ---------------------------------------------------------------------------
# Tokenization (integration-level)
# ---------------------------------------------------------------------------

class TestTokenizeDataset:
    @pytest.fixture(scope="class")
    def tokenized(self):
        from src.config import DataConfig, ModelConfig
        from src.preprocess import build_tokenizer, tokenize_dataset

        rows = [
            {"user_input": "This is fine.", "toxicity": 0},
            {"user_input": "You are terrible!", "toxicity": 1},
            {"user_input": "Hello world.", "toxicity": 0},
        ]
        dd = _make_dataset(rows)
        model_cfg = ModelConfig(name="distilbert-base-uncased", max_length=32)
        data_cfg = DataConfig(text_field="user_input", label_field="toxicity", normalize_text=True)
        tokenizer = build_tokenizer(model_cfg)
        return tokenize_dataset(dd, tokenizer, data_cfg, model_cfg)

    def test_has_input_ids(self, tokenized):
        assert "input_ids" in tokenized["train"].column_names

    def test_has_attention_mask(self, tokenized):
        assert "attention_mask" in tokenized["train"].column_names

    def test_has_labels(self, tokenized):
        assert "labels" in tokenized["train"].column_names

    def test_no_raw_text_column(self, tokenized):
        assert "user_input" not in tokenized["train"].column_names

    def test_correct_row_count(self, tokenized):
        assert len(tokenized["train"]) == 3

    def test_input_ids_length(self, tokenized):
        ids = tokenized["train"]["input_ids"]
        assert all(len(row) == 32 for row in ids)


# ---------------------------------------------------------------------------
# Threshold metric computation (used in evaluate_model)
# ---------------------------------------------------------------------------

class TestThresholdMetrics:
    def test_perfect_predictions(self):
        from src.evaluate_model import _metrics_at_threshold
        import numpy as np

        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        m = _metrics_at_threshold(labels, probs, threshold=0.5)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["accuracy"] == 1.0

    def test_all_wrong(self):
        from src.evaluate_model import _metrics_at_threshold
        import numpy as np

        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.9, 0.9, 0.1, 0.1])
        m = _metrics_at_threshold(labels, probs, threshold=0.5)
        assert m["tp"] == 0
        assert m["tn"] == 0

    def test_low_threshold_increases_recall(self):
        from src.evaluate_model import _metrics_at_threshold
        import numpy as np

        labels = np.array([0, 0, 1, 1, 1])
        probs = np.array([0.2, 0.4, 0.3, 0.6, 0.8])

        m_low = _metrics_at_threshold(labels, probs, threshold=0.2)
        m_high = _metrics_at_threshold(labels, probs, threshold=0.7)
        assert m_low["recall"] >= m_high["recall"]

    def test_high_threshold_increases_precision(self):
        from src.evaluate_model import _metrics_at_threshold
        import numpy as np

        labels = np.array([0, 0, 1, 1, 1])
        probs = np.array([0.2, 0.4, 0.3, 0.6, 0.8])

        m_low = _metrics_at_threshold(labels, probs, threshold=0.2)
        m_high = _metrics_at_threshold(labels, probs, threshold=0.7)
        assert m_high["precision"] >= m_low["precision"]

    def test_threshold_stored_in_result(self):
        from src.evaluate_model import _metrics_at_threshold
        import numpy as np

        labels = np.array([0, 1])
        probs = np.array([0.2, 0.8])
        m = _metrics_at_threshold(labels, probs, threshold=0.3)
        assert m["threshold"] == 0.3
