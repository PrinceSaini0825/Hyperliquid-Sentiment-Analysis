"""
Model Validation Module
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def validate_model(model, X_test, y_test):
    """
    Validate trained model and save metrics.
    """

    print("\nValidating Model...")

    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
    }

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
        metrics["ROC_AUC"] = roc_auc_score(
            y_test,
            prob
        )

    print("\nValidation Results")
    print("-" * 40)

    for k, v in metrics.items():
        print(f"{k:<12}: {v:.4f}")

    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "validation_metrics.csv",
        index=False
    )

    print("✓ Validation Complete")

    return metrics