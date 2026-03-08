"""
preprocess.py — Feature engineering and transformation pipeline
Responsibilities:
  1. Load raw data from data/raw/heart.csv
  2. Split into train/test sets (reproducible via random_seed from config)
  3. Build sklearn Pipeline with separate numeric and categorical transformers
  4. Save processed feature snapshot to data/processed/features.parquet
  5. Return X_train, X_test, y_train, y_test for use by train.py
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        # Always resolve relative to project root, regardless of where script is run from
        root = Path(__file__).resolve().parent.parent
        config_path = root / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info(f"Loaded raw data: {df.shape}")
    return df


def split_features_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    log.info(f"Features: {X.shape[1]} columns | Target: {y.value_counts().to_dict()}")
    return X, y


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies different transformations
    to numeric vs categorical columns independently.

    Numeric pipeline:
      - SimpleImputer(median): fills missing values with column median
        (median is more robust than mean for skewed medical data)
      - StandardScaler: zero mean, unit variance
        (required for distance-based models; good practice universally)

    Categorical pipeline:
      - SimpleImputer(most_frequent): fills missing with most common value
      - OneHotEncoder: converts integers like 1/2/3/4 into binary columns
        (prevents model from assuming 4 > 3 > 2 > 1 ordering)
    """

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ])

    return preprocessor


def save_processed_snapshot(X_train, X_test, y_train, y_test, cfg: dict) -> None:
    """
    Save a feature snapshot to parquet format.
    Parquet is the industry standard for feature storage:
    - Columnar format = fast reads
    - Preserves dtypes
    - Much smaller than CSV
    """
    processed_path = cfg["data"]["processed_path"]
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)

    # Reconstruct a labelled DataFrame for the snapshot
    train_df = pd.DataFrame(X_train).assign(split="train", target=y_train.values)
    test_df  = pd.DataFrame(X_test).assign(split="test",  target=y_test.values)
    snapshot = pd.concat([train_df, test_df], ignore_index=True)

    snapshot.to_parquet(processed_path, index=False)
    log.info(f"Feature snapshot saved to: {processed_path} | shape: {snapshot.shape}")


def run_preprocessing():
    log.info("=== Starting preprocessing ===")
    cfg = load_config()

    df = load_raw_data(cfg["data"]["raw_path"])

    X, y = split_features_target(df, cfg["features"]["target"])

    # Reproducible split — seed comes from config, never hardcoded
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["project"]["random_seed"],
        stratify=y   # ensures class ratio is preserved in both splits
    )
    log.info(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    preprocessor = build_preprocessor(
        numeric_features=cfg["features"]["numeric"],
        categorical_features=cfg["features"]["categorical"],
    )

    # Fit on train only — NEVER fit on test data
    # This is one of the most common data leakage mistakes
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)
    log.info(f"Processed feature shape — Train: {X_train_processed.shape} | Test: {X_test_processed.shape}")

    save_processed_snapshot(X_train_processed, X_test_processed, y_train, y_test, cfg)

    log.info("=== Preprocessing complete ===")
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


if __name__ == "__main__":
    run_preprocessing()