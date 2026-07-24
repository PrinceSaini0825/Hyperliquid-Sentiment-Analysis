"""
Experiment Tracking
"""

from pathlib import Path
from datetime import datetime
import json

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def save_experiment(
    model_name,
    metrics,
    feature_count,
    train_size,
    test_size
):
    """
    Save experiment metadata.
    """

    experiment = {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "model": model_name,
        "train_size": train_size,
        "test_size": test_size,
        "feature_count": feature_count,
        "accuracy": float(metrics["Accuracy"]),
        "precision": float(metrics["Precision"]),
        "recall": float(metrics["Recall"]),
        "f1_score": float(metrics["F1"]),
        "roc_auc": float(metrics["ROC_AUC"])
    }

    filename = (
        LOG_DIR /
        f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(filename, "w") as f:
        json.dump(
            experiment,
            f,
            indent=4
        )

    print("✓ Experiment Saved")