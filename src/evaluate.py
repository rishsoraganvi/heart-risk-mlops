"""
evaluate.py — Metrics computation and visualization
Kept separate from train.py so evaluation logic can be reused
independently (e.g. during retraining comparison).
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

log = logging.getLogger(__name__)


def compute_metrics(pipeline, X_test, y_test) -> dict:
    """Compute all evaluation metrics. Returns dict for MLflow logging."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return {
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
    }


def save_confusion_matrix(pipeline, X_test, y_test, path: str = "confusion_matrix.png") -> str:
    """Save confusion matrix plot and return file path for MLflow artifact logging."""
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No Disease", "Disease"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    log.info(f"Confusion matrix saved to: {path}")
    return path