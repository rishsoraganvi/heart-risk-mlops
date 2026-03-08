"""
ingest.py — Automated ETL entrypoint
Responsibilities:
  1. Download raw data from source URL (config-driven)
  2. Assign column names (dataset has no header)
  3. Handle missing values marked as '?'
  4. Validate schema and basic quality checks
  5. Save to data/raw/heart.csv
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import requests
import yaml

# ── Logging setup ─────────────────────────────────────────────
# Industry standard: structured logs, not print statements.
# Every log line shows timestamp + level + message.
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


def download_data(url: str, save_path: str) -> pd.DataFrame:
    """Download CSV from URL, assign column names, handle missing values."""

    # Column names for UCI Heart Disease dataset (no header in source file)
    columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal", "target"
    ]

    log.info(f"Downloading data from: {url}")
    response = requests.get(url, timeout=30)

    # Fail immediately if download failed — never proceed silently
    if response.status_code != 200:
        log.error(f"Download failed with status code: {response.status_code}")
        sys.exit(1)

    # Save raw response to disk first
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(response.text)

    # Read back as DataFrame
    df = pd.read_csv(
        save_path,
        header=None,
        names=columns,
        na_values="?"   # UCI dataset uses '?' for missing values
    )

    log.info(f"Downloaded {len(df)} rows, {len(df.columns)} columns")
    return df


def validate_data(df: pd.DataFrame, cfg: dict) -> None:
    """
    Schema + quality validation.
    Raises ValueError immediately if data doesn't meet expectations.
    This is the professional pattern: fail fast, fail loudly.
    """
    expected_cols = cfg["features"]["numeric"] + cfg["features"]["categorical"] + [cfg["features"]["target"]]

    # Check all expected columns exist
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Check dataset isn't empty
    if len(df) == 0:
        raise ValueError("Dataset is empty after download")

    # Check target column exists and has valid values
    if df["target"].isnull().all():
        raise ValueError("Target column is entirely null")

    # Log missing value summary — important to know before preprocessing
    null_counts = df.isnull().sum()
    nulls_present = null_counts[null_counts > 0]
    if not nulls_present.empty:
        log.warning(f"Missing values detected:\n{nulls_present}")
    else:
        log.info("No missing values found")

    log.info("Schema validation passed ✓")


def save_data(df: pd.DataFrame, save_path: str) -> None:
    """Save validated DataFrame — overwrite safe with explicit path."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    log.info(f"Data saved to: {save_path}")


def run_ingest():
    log.info("=== Starting data ingestion ===")
    cfg = load_config()

    df = download_data(
        url=cfg["data"]["source_url"],
        save_path=cfg["data"]["raw_path"]
    )

    validate_data(df, cfg)

    # Binarize target: original has values 0-4, we want 0 (no disease) vs 1 (disease)
    df["target"] = (df["target"] > 0).astype(int)
    log.info(f"Target distribution:\n{df['target'].value_counts()}")

    save_data(df, cfg["data"]["raw_path"])
    log.info("=== Ingestion complete ===")


if __name__ == "__main__":
    run_ingest()