"""
ROC Curve Visualization Module
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def save_roc_curve(model, X_test, y_test):
    """
    Generate and save ROC curve.
    """

    print("\nGenerating ROC Curve...")

    # Probability predictions
    if not hasattr(model, "predict_proba"):
        print(
            "Model does not support probability prediction."
        )
        return


    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # ROC values
    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )


    # AUC score
    auc_score = roc_auc_score(
        y_test,
        probabilities
    )


    # Plot ROC curve
    plt.figure(
        figsize=(8, 6)
    )


    plt.plot(
        fpr,
        tpr,
        label=f"ROC Curve (AUC = {auc_score:.4f})"
    )


    # Random classifier baseline
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Receiver Operating Characteristic (ROC) Curve"
    )


    plt.legend(
        loc="lower right"
    )


    plt.grid(
        True
    )


    plt.savefig(
        RESULTS_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"✓ ROC Curve Saved (AUC={auc_score:.4f})"
    )


    return auc_score