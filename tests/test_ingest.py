"""Unit tests for ingest.py"""
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ingest import validate_data


@pytest.fixture
def sample_config():
    return {
        "features": {
            "numeric": ["age", "trestbps", "chol", "thalach", "oldpeak"],
            "categorical": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
            "target": "target"
        }
    }


@pytest.fixture
def valid_df():
    cols = ["age","trestbps","chol","thalach","oldpeak",
            "sex","cp","fbs","restecg","exang","slope","ca","thal","target"]
    return pd.DataFrame([[63,145,233,150,2.3,1,1,1,2,0,3,0,6,0]], columns=cols)


def test_validate_passes_on_valid_data(valid_df, sample_config):
    """Should not raise on clean data"""
    validate_data(valid_df, sample_config)


def test_validate_fails_on_missing_column(valid_df, sample_config):
    """Should raise ValueError when a required column is missing"""
    df_bad = valid_df.drop(columns=["age"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_data(df_bad, sample_config)


def test_validate_fails_on_empty_dataframe(sample_config):
    """Should raise ValueError on empty DataFrame"""
    cols = ["age","trestbps","chol","thalach","oldpeak",
            "sex","cp","fbs","restecg","exang","slope","ca","thal","target"]
    df_empty = pd.DataFrame(columns=cols)
    with pytest.raises(ValueError, match="empty"):
        validate_data(df_empty, sample_config)