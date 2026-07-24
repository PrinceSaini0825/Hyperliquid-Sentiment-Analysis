"""
Confusion Matrix Visualization Module
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)



def save_confusion_matrix(
    model,
    X_test,
    y_test
):
    """
    Generate and save confusion matrix.
    """

    print("\nGenerating Confusion Matrix...")


    # Predictions
    predictions = model.predict(
        X_test
    )


    # Matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )


    print("\nConfusion Matrix")
    print("----------------")
    print(cm)


    # Save raw values
    cm_df = pd.DataFrame(
        cm,
        columns=[
            "Predicted Negative",
            "Predicted Positive"
        ],
        index=[
            "Actual Negative",
            "Actual Positive"
        ]
    )


    cm_df.to_csv(
        RESULTS_DIR /
        "confusion_matrix.csv"
    )


    # Plot
    plt.figure(
        figsize=(7, 6)
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Loss",
            "Profit"
        ]
    )


    display.plot()


    plt.title(
        "Confusion Matrix - Profit Prediction"
    )


    plt.tight_layout()


    plt.savefig(
        RESULTS_DIR /
        "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



    # Classification report
    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Loss",
            "Profit"
        ],
        output_dict=True
    )


    report_df = pd.DataFrame(
        report
    ).transpose()


    report_df.to_csv(
        RESULTS_DIR /
        "classification_report.csv"
    )


    print(
        "✓ Confusion Matrix Saved"
    )


    return cm