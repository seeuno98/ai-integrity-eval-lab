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
    """Test the internal _clean_split helper via tokenize_dataset."""

    def _run(self, rows, text_field="text", label_field="toxicity"):
        from src.preprocess import _clean_split

        dd = _make_dataset(rows)
        return _clean_split(dd, text_field, label_field)

    def test_removes_null_text(self):
        rows = [
            {"text": None, "toxicity": 0},
            {"text": "hello", "toxicity": 0},
        ]
        result = self._run(rows)
        assert len(result["train"]) == 1

    def test_removes_empty_text(self):
        rows = [
            {"text": "   ", "toxicity": 0},
            {"text": "valid text", "toxicity": 1},
        ]
        result = self._run(rows)
        assert len(result["train"]) == 1

    def test_removes_null_label(self):
        rows = [
            {"text": "ok text", "toxicity": None},
            {"text": "valid", "toxicity": 1},
        ]
        result = self._run(rows)
        assert len(result["train"]) == 1

    def test_keeps_valid_rows(self):
        rows = [
            {"text": "first", "toxicity": 0},
            {"text": "second", "toxicity": 1},
        ]
        result = self._run(rows)
        assert len(result["train"]) == 2


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
        labels = result["train"]["labels"]
        assert list(labels) == [0, 1]

    def test_original_label_column_removed(self):
        rows = [{"text": "a", "toxicity": 0}]
        result = self._run(rows)
        assert "toxicity" not in result["train"].column_names

    def test_unexpected_label_raises(self):
        from src.preprocess import _map_labels

        rows = [{"text": "a", "toxicity": 99}]
        dd = _make_dataset(rows)
        with pytest.raises(ValueError, match="Unexpected label"):
            _map_labels(dd, "toxicity")


# ---------------------------------------------------------------------------
# Tokenization (integration-level)
# ---------------------------------------------------------------------------

class TestTokenizeDataset:
    """Test that tokenize_dataset produces the expected column layout."""

    @pytest.fixture(scope="class")
    def tokenized(self):
        from src.config import DataConfig, ModelConfig
        from src.preprocess import build_tokenizer, tokenize_dataset

        rows = [
            {"user_prompt": "This is fine.", "toxicity": 0},
            {"user_prompt": "You are terrible!", "toxicity": 1},
            {"user_prompt": "Hello world.", "toxicity": 0},
        ]
        dd = _make_dataset(rows)

        model_cfg = ModelConfig(name="distilbert-base-uncased", max_length=32)
        data_cfg = DataConfig(text_field="user_prompt", label_field="toxicity")
        tokenizer = build_tokenizer(model_cfg)
        return tokenize_dataset(dd, tokenizer, data_cfg, model_cfg)

    def test_has_input_ids(self, tokenized):
        assert "input_ids" in tokenized["train"].column_names

    def test_has_attention_mask(self, tokenized):
        assert "attention_mask" in tokenized["train"].column_names

    def test_has_labels(self, tokenized):
        assert "labels" in tokenized["train"].column_names

    def test_no_raw_text_column(self, tokenized):
        assert "user_prompt" not in tokenized["train"].column_names

    def test_correct_row_count(self, tokenized):
        assert len(tokenized["train"]) == 3

    def test_input_ids_length(self, tokenized):
        ids = tokenized["train"]["input_ids"]
        # Each row should be padded/truncated to max_length=32
        assert all(len(row) == 32 for row in ids)
