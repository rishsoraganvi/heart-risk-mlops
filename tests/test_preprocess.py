"""Unit tests for preprocess.py"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocess import build_preprocessor, split_features_target


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [63.0, 37.0, 41.0, 56.0],
        "trestbps": [145.0, 130.0, 130.0, 120.0],
        "chol": [233.0, 250.0, 204.0, 236.0],
        "thalach": [150.0, 187.0, 172.0, 178.0],
        "oldpeak": [2.3, 3.5, 1.4, 0.8],
        "sex": [1.0, 1.0, 0.0, 1.0],
        "cp": [1.0, 2.0, 1.0, 2.0],
        "fbs": [1.0, 0.0, 0.0, 0.0],
        "restecg": [2.0, 0.0, 2.0, 2.0],
        "exang": [0.0, 0.0, 0.0, 0.0],
        "slope": [3.0, 2.0, 1.0, 1.0],
        "ca": [0.0, 0.0, 0.0, 0.0],
        "thal": [6.0, 3.0, 3.0, 3.0],
        "target": [0, 0, 0, 0],
    })


def test_split_features_target(sample_df):
    """Target column should be separated correctly"""
    X, y = split_features_target(sample_df, "target")
    assert "target" not in X.columns
    assert len(y) == len(sample_df)
    assert list(y) == [0, 0, 0, 0]


def test_preprocessor_output_shape(sample_df):
    """Preprocessor should expand categorical columns via OHE"""
    numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    X, _ = split_features_target(sample_df, "target")

    preprocessor = build_preprocessor(numeric, categorical)
    X_transformed = preprocessor.fit_transform(X)

    # Should have more columns than input due to OHE expansion
    assert X_transformed.shape[0] == 4
    assert X_transformed.shape[1] > len(numeric) + len(categorical)


def test_preprocessor_no_nulls_after_transform(sample_df):
    """No NaN values should remain after preprocessing"""
    # Introduce a missing value
    sample_df.loc[0, "ca"] = np.nan
    numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    X, _ = split_features_target(sample_df, "target")

    preprocessor = build_preprocessor(numeric, categorical)
    X_transformed = preprocessor.fit_transform(X)

    assert not np.isnan(X_transformed).any()