"""
batch_infer.py — Batch inference using registered MLflow model
Responsibilities:
  1. Load the latest production model from MLflow Model Registry
  2. Accept new raw patient data (same format as training data)
  3. Run predictions with confidence scores
  4. Save results to outputs/predictions.csv
"""

import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        root = Path(__file__).resolve().parent.parent
        config_path = root / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_name: str, tracking_uri: str, version: int = None):
    """
    Load model from MLflow Model Registry.

    If version is None, loads the LATEST version automatically.
    This is the professional pattern — you never hardcode a model path.
    The registry is the single source of truth for which model is active.
    """
    mlflow.set_tracking_uri(tracking_uri)

    if version:
        model_uri = f"models:/{model_name}/{version}"
    else:
        model_uri = f"models:/{model_name}/latest"

    log.info(f"Loading model from registry: {model_uri}")
    pipeline = mlflow.sklearn.load_model(model_uri)
    log.info("Model loaded successfully")
    return pipeline


def prepare_input(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Validate incoming data has required feature columns.
    In production this would be a strict schema check.
    """
    expected = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    missing = set(expected) - set(data.columns)
    if missing:
        raise ValueError(f"Input data missing columns: {missing}")
    return data[expected]   # return only feature columns in correct order


def run_batch_inference(input_path: str = None, version: int = None):
    log.info("=== Starting batch inference ===")
    cfg = load_config()
    root = Path(__file__).resolve().parent.parent

    # ── Load model from registry ──────────────────────────────
    tracking_uri = (root / cfg["mlflow"]["tracking_uri"]).as_uri()
    pipeline = load_model(
        model_name=cfg["mlflow"]["model_name"],
        tracking_uri=tracking_uri,
        version=version,
    )

    # ── Load input data ───────────────────────────────────────
    # Default: reuse raw data to simulate new patient batch
    if input_path is None:
        input_path = str(root / cfg["data"]["raw_path"])
        log.info(f"No input file specified — using: {input_path}")

    df_input = pd.read_csv(input_path)
    log.info(f"Input batch: {len(df_input)} records")

    # ── Prepare features ──────────────────────────────────────
    X = prepare_input(df_input, cfg)

    # ── Predict ───────────────────────────────────────────────
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]  # probability of disease

    # ── Build output DataFrame ────────────────────────────────
    results = df_input.copy()
    results["predicted_risk"] = predictions
    results["risk_probability"] = probabilities.round(4)
    results["risk_label"] = results["predicted_risk"].map({
        0: "Low Risk",
        1: "High Risk"
    })

    # ── Save predictions ──────────────────────────────────────
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "predictions.csv"
    results.to_csv(output_path, index=False)

    # Summary stats
    high_risk = results["predicted_risk"].sum()
    low_risk = len(results) - high_risk
    log.info(f"Predictions complete — High Risk: {high_risk} | Low Risk: {low_risk}")
    log.info(f"Results saved to: {output_path}")
    log.info("=== Batch inference complete ===")

    return results


if __name__ == "__main__":
    run_batch_inference()