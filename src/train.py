"""
train.py — MLflow-tracked model training entrypoint
Responsibilities:
  1. Load config (all hyperparams come from here)
  2. Run preprocessing pipeline
  3. Train RandomForestClassifier inside an MLflow run
  4. Log params, metrics, and model artifact to MLflow
  5. Register model in MLflow Model Registry
"""

import logging

import mlflow
import mlflow.sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor, load_raw_data, split_features_target
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def build_full_pipeline(preprocessor, model_params: dict) -> Pipeline:
    """
    Chain preprocessor + model into a single sklearn Pipeline.
    Industry benefit: the pipeline object contains EVERYTHING needed
    to go from raw input → prediction. One object to save, one to load.
    """
    model = RandomForestClassifier(
        **model_params,
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def log_metrics(metrics: dict) -> None:
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
        log.info(f"  {name}: {value:.4f}")


def run_training():
    log.info("=== Starting training ===")
    cfg = load_config()
    root = Path(__file__).resolve().parent.parent
    raw_path = str(root / cfg["data"]["raw_path"])

    # ── Data preparation ──────────────────────────────────────
    df = load_raw_data(raw_path)
    X, y = split_features_target(df, cfg["features"]["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["project"]["random_seed"],
        stratify=y,
    )

    preprocessor = build_preprocessor(
        numeric_features=cfg["features"]["numeric"],
        categorical_features=cfg["features"]["categorical"],
    )

    # ── MLflow experiment setup ───────────────────────────────
    # All runs go into a named experiment — not the default bucket
    root = Path(__file__).resolve().parent.parent
    mlflow.set_tracking_uri((root / cfg["mlflow"]["tracking_uri"]).as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # ── Training run ──────────────────────────────────────────
    # Everything inside this block is tracked automatically
    with mlflow.start_run() as run:
        log.info(f"MLflow Run ID: {run.info.run_id}")

        # Log all hyperparams from config — full reproducibility
        model_params = cfg["model"]["params"]
        model_params["random_state"] = cfg["project"]["random_seed"]
        mlflow.log_params(model_params)
        mlflow.log_param("model_name", cfg["model"]["name"])
        mlflow.log_param("test_size", cfg["data"]["test_size"])
        mlflow.log_param("train_rows", len(X_train))

        # Build and train full pipeline
        pipeline = build_full_pipeline(preprocessor, model_params)
        pipeline.fit(X_train, y_train)
        log.info("Model training complete")

        # ── Evaluation ────────────────────────────────────────
        from evaluate import compute_metrics, save_confusion_matrix
        metrics = compute_metrics(pipeline, X_test, y_test)

        log.info("Metrics:")
        log_metrics(metrics)

        # Log confusion matrix plot as artifact
        cm_path = save_confusion_matrix(pipeline, X_test, y_test)
        mlflow.log_artifact(cm_path)

        # ── Model registration ────────────────────────────────
        # Log full pipeline (preprocessor + model) as one artifact
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=cfg["mlflow"]["model_name"],
        )
        log.info(f"Model registered as: {cfg['mlflow']['model_name']}")

    log.info(f"=== Training complete | Run ID: {run.info.run_id} ===")
    return run.info.run_id


if __name__ == "__main__":
    run_training()