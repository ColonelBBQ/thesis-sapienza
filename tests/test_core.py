from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from code.metrics import (
    build_prediction_arrays,
    compute_per_label_accuracies,
    compute_precision_recall_from_confusion_df,
)
from code.bert_pipeline import (
    DEFAULT_TARGET_LIST,
    BertExperimentConfig,
    filter_subset,
    prepare_baseline_dataframe,
    set_seed,
)
from code.bert_experiments import sanitize_experiment_name


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "description": ["fast system", "secure platform", "scalable service"],
        "source": ["doc1", "doc2", "doc3"],
        "goal": ["g1", "g2", "g3"],
        "ambuiguity": [1, 0, 0],
        "ambiguity_type": ["lexical", "", ""],
        "compute": [1, 0, 0],
        "data_handling": [0, 1, 0],
        "network": [0, 0, 1],
        "security_compliance": [0, 1, 0],
        "management_monitoring": [0, 0, 0],
        "cloud_service_essentials": [0, 0, 0],
    })


class TestBertExperimentConfig:
    def test_defaults(self) -> None:
        config = BertExperimentConfig()
        assert config.threshold == 0.20
        assert config.dropout == 0.3
        assert config.num_workers == 0
        assert config.model_name == "bert-base-uncased"

    def test_custom_values(self) -> None:
        config = BertExperimentConfig(threshold=0.5, epochs=10, seed=123)
        assert config.threshold == 0.5
        assert config.epochs == 10

    def test_is_frozen(self) -> None:
        config = BertExperimentConfig()
        with pytest.raises(Exception):
            config.epochs = 10  # type: ignore[misc]


class TestPrepareBaselineDataframe:
    def test_drops_default_columns(self, sample_df: pd.DataFrame) -> None:
        result = prepare_baseline_dataframe(sample_df)
        for col in ["source", "goal", "ambiguity_type", "ambuiguity"]:
            assert col not in result.columns

    def test_targets_are_int(self, sample_df: pd.DataFrame) -> None:
        result = prepare_baseline_dataframe(sample_df)
        for label in DEFAULT_TARGET_LIST:
            assert str(result[label].dtype).startswith("int")


class TestFilterSubset:
    def test_query_filter(self, sample_df: pd.DataFrame) -> None:
        result = filter_subset(sample_df, query="compute == 1", subset_name="compute_only")
        assert len(result) == 1

    def test_mask_filter(self, sample_df: pd.DataFrame) -> None:
        mask = sample_df["ambuiguity"] == 1
        result = filter_subset(sample_df, mask=mask)
        assert len(result) == 1


class TestMetrics:
    def test_build_prediction_arrays(self) -> None:
        targets = [[1, 0], [0, 1]]
        probs = [[0.9, 0.1], [0.3, 0.8]]
        t, p, o = build_prediction_arrays(targets, probs, 0.5)
        assert t.shape == (2, 2)
        assert o[0, 0] == 1
        assert o[0, 1] == 0

    def test_compute_per_label_accuracies(self) -> None:
        targets = np.array([[1, 0], [0, 1]])
        outputs = np.array([[1, 0], [0, 1]])
        acc = compute_per_label_accuracies(targets, outputs, ["a", "b"])
        assert acc == {"a": 1.0, "b": 1.0}

    def test_precision_recall_perfect(self) -> None:
        result = compute_precision_recall_from_confusion_df(10.0, 0.0, 0.0)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_precision_recall_zero(self) -> None:
        result = compute_precision_recall_from_confusion_df(0.0, 0.0, 10.0)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0


class TestSetSeed:
    def test_reproducibility(self) -> None:
        set_seed(42)
        a = np.random.rand(3).copy()
        set_seed(42)
        b = np.random.rand(3)
        assert np.array_equal(a, b)


class TestSanitizeExperimentName:
    def test_spaces_replaced(self) -> None:
        assert sanitize_experiment_name("my experiment v1") == "my_experiment_v1"

    def test_special_chars(self) -> None:
        assert sanitize_experiment_name("test!@#name") == "test_name"

    def test_empty_fallback(self) -> None:
        assert sanitize_experiment_name("!!!") == "experiment"
