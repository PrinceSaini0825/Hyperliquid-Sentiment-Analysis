"""
Model Evaluation Module
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model and generate reports.
    """

    print("\nEvaluating Best Model...")

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
    }

    print("\nEvaluation Metrics")
    print("-" * 40)

    for k, v in metrics.items():
        print(f"{k:12}: {v:.4f}")

    pd.DataFrame([metrics]).to_csv(
        REPORT_DIR / "evaluation_metrics.csv",
        index=False
    )

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    with open(
        REPORT_DIR / "classification_report.txt",
        "w"
    ) as f:
        f.write(report)

    # ------------------------
    # Confusion Matrix
    # ------------------------

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)

    plt.title("Confusion Matrix")

    plt.colorbar()

    plt.xticks([0, 1], ["Loss", "Profit"])
    plt.yticks([0, 1], ["Loss", "Profit"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ------------------------
    # ROC Curve
    # ------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    plt.figure(figsize=(6, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC={metrics['ROC_AUC']:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "roc_curve.png",
        dpi=300
    )

    plt.close()

    # ------------------------
    # Precision Recall Curve
    # ------------------------

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    plt.figure(figsize=(6, 5))

    plt.plot(
        recall,
        precision
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision Recall Curve")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "precision_recall_curve.png",
        dpi=300
    )

    plt.close()

    print("✓ Evaluation Complete")