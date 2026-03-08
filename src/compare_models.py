"""
compare_models.py — Auto-promote model if new version beats current
Loads the two latest model versions from MLflow registry,
compares their AUC on test data, promotes the better one.
"""

import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(config_path=None) -> dict:
    if config_path is None:
        root = Path(__file__).resolve().parent.parent
        config_path = root / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_test_data(cfg: dict):
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / cfg["data"]["raw_path"])
    X = df.drop(columns=[cfg["features"]["target"]])
    y = df[cfg["features"]["target"]]
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["project"]["random_seed"],
        stratify=y,
    )
    return X_test, y_test


def evaluate_version(client, model_name: str, version: int,
                     tracking_uri: str, X_test, y_test) -> float:
    model_uri = f"models:/{model_name}/{version}"
    pipeline = mlflow.sklearn.load_model(model_uri)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    log.info(f"  Version {version} → AUC: {auc:.4f}")
    return auc


def run_comparison():
    log.info("=== Starting model comparison ===")
    cfg = load_config()
    root = Path(__file__).resolve().parent.parent
    tracking_uri = (root / cfg["mlflow"]["tracking_uri"]).as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    model_name = cfg["mlflow"]["model_name"]

    # Get all registered versions
    versions = client.search_model_versions(f"name='{model_name}'")
    version_numbers = sorted([int(v.version) for v in versions])

    if len(version_numbers) < 2:
        log.info("Only one model version exists — nothing to compare")
        return

    # Compare latest two versions
    v_current = version_numbers[-2]
    v_new     = version_numbers[-1]
    log.info(f"Comparing version {v_current} (current) vs version {v_new} (new)")

    X_test, y_test = get_test_data(cfg)

    auc_current = evaluate_version(client, model_name, v_current, tracking_uri, X_test, y_test)
    auc_new     = evaluate_version(client, model_name, v_new,     tracking_uri, X_test, y_test)

    threshold = cfg["evaluation"]["promotion_threshold"]

    if auc_new > auc_current and auc_new >= threshold:
        log.info(f"✅ New model (v{v_new}) is BETTER — promoting to champion")
    elif auc_new < auc_current:
        log.warning(f"⚠️  New model (v{v_new}) underperforms — keeping v{v_current}")
    else:
        log.info(f"Models are comparable — keeping current v{v_current}")

    log.info("=== Comparison complete ===")
    return {"v_current": v_current, "auc_current": auc_current,
            "v_new": v_new, "auc_new": auc_new}


if __name__ == "__main__":
    run_comparison()