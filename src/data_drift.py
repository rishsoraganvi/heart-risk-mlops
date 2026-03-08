"""
data_drift.py — Feature distribution drift detection using PSI
PSI (Population Stability Index) is the industry standard for
detecting when incoming data has shifted from training data.

PSI interpretation:
  < 0.10  → No significant drift, model is stable
  0.10 – 0.25 → Moderate drift, monitor closely
  > 0.25  → Significant drift, retrain immediately
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
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


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute PSI between expected (training) and actual (new) distributions.

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    We add a small epsilon to avoid log(0) which would blow up the calculation.
    """
    epsilon = 1e-8

    # Create bin edges from expected distribution
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )

    # Bin both distributions
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    # Convert to proportions
    expected_pct = expected_counts / len(expected) + epsilon
    actual_pct   = actual_counts   / len(actual)   + epsilon

    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 6)


def run_drift_check(new_data_path: str = None) -> dict:
    """
    Compare feature distributions between training data and new data.
    Returns dict of PSI scores per feature + overall drift flag.
    """
    log.info("=== Starting drift detection ===")
    cfg = load_config()
    root = Path(__file__).resolve().parent.parent

    # Load training reference data
    train_path = root / cfg["data"]["raw_path"]
    train_df = pd.read_csv(train_path)
    log.info(f"Reference (training) data: {train_df.shape}")

    # Load new data — simulate drift by adding Gaussian noise to training data
    # In production this would be your actual new incoming data
    if new_data_path is None:
        log.info("No new data path provided — simulating drift with noise")
        new_df = train_df.copy()
        numeric_cols = cfg["features"]["numeric"]
        # Add noise to simulate real-world distribution shift
        for col in numeric_cols:
            noise = np.random.normal(0, new_df[col].std() * 0.3, size=len(new_df))
            new_df[col] = new_df[col] + noise
    else:
        new_df = pd.read_csv(new_data_path)

    log.info(f"New data: {new_df.shape}")

    # ── Compute PSI for each numeric feature ──────────────────
    threshold = cfg["retraining"]["drift_threshold"]
    numeric_features = cfg["features"]["numeric"]
    psi_scores = {}
    drifted_features = []

    for feature in numeric_features:
        if feature in train_df.columns and feature in new_df.columns:
            psi = compute_psi(
                train_df[feature].dropna().values,
                new_df[feature].dropna().values,
            )
            psi_scores[feature] = psi
            status = "⚠️  DRIFT" if psi > threshold else "✅ stable"
            log.info(f"  {feature:<12} PSI={psi:.4f}  {status}")
            if psi > threshold:
                drifted_features.append(feature)

    # ── Overall drift decision ────────────────────────────────
    mean_psi = np.mean(list(psi_scores.values()))
    drift_detected = mean_psi > threshold

    log.info(f"Mean PSI across features: {mean_psi:.4f}")
    if drift_detected:
        log.warning(f"DRIFT DETECTED — {len(drifted_features)} features drifted: {drifted_features}")
        log.warning("Recommendation: trigger retraining pipeline")
    else:
        log.info("No significant drift detected — model remains valid")

    log.info("=== Drift detection complete ===")

    return {
        "psi_scores": psi_scores,
        "mean_psi": mean_psi,
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
    }


if __name__ == "__main__":
    run_drift_check()