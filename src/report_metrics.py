"""
report_metrics.py — Extract best run metrics and write to metrics_report.json
This file is committed to Git so metrics are visible in the repo
without needing to run MLflow UI.
"""

import json
import logging
from pathlib import Path

import mlflow
import yaml

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


def get_best_run() -> dict:
    cfg = load_config()
    root = Path(__file__).resolve().parent.parent
    tracking_uri = (root / cfg["mlflow"]["tracking_uri"]).as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
    if not experiment:
        log.error("No experiment found")
        return {}

    # Get all runs sorted by AUC descending
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )

    if not runs:
        log.error("No runs found")
        return {}

    best = runs[0]
    result = {
        "run_id": best.info.run_id,
        "roc_auc":   round(best.data.metrics.get("roc_auc", 0), 4),
        "accuracy":  round(best.data.metrics.get("accuracy", 0), 4),
        "f1":        round(best.data.metrics.get("f1", 0), 4),
        "recall":    round(best.data.metrics.get("recall", 0), 4),
        "precision": round(best.data.metrics.get("precision", 0), 4),
        "params":    best.data.params,
    }

    log.info(f"Best run: {result['run_id']}")
    log.info(f"  AUC: {result['roc_auc']} | F1: {result['f1']} | Recall: {result['recall']}")
    return result


def run_report():
    log.info("=== Generating metrics report ===")
    best = get_best_run()
    if not best:
        return

    root = Path(__file__).resolve().parent.parent
    report_path = root / "metrics_report.json"

    with open(report_path, "w") as f:
        json.dump(best, f, indent=2)

    log.info(f"Report written to: {report_path}")
    log.info("=== Report complete ===")


if __name__ == "__main__":
    run_report()