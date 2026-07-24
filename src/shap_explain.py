"""
SHAP Explainability Module

Fast TreeSHAP implementation
"""

from pathlib import Path

import shap
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def generate_shap_analysis(
    model,
    X_test,
    feature_names=None,
    sample_size=500
):

    print("\nGenerating SHAP Explainability...")


    # Sample data
    if len(X_test) > sample_size:
        X_sample = X_test.sample(
            sample_size,
            random_state=42
        )
    else:
        X_sample = X_test.copy()


    if feature_names:
        X_sample.columns = feature_names


    print(
        f"SHAP samples: {len(X_sample)}"
    )


    # Fast Tree Explainer
    explainer = shap.TreeExplainer(
        model
    )


    print("Calculating SHAP values...")


    shap_values = explainer(
        X_sample
    )


    # New SHAP API
    values = shap_values.values


    # Binary classifier handling
    if len(values.shape) == 3:
        values = values[:,:,1]


    print("Creating SHAP plot...")


    plt.figure(
        figsize=(10,8)
    )


    shap.summary_plot(
        values,
        X_sample,
        show=False
    )


    plt.tight_layout()


    plt.savefig(
        RESULTS_DIR /
        "shap_summary_plot.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    importance = pd.DataFrame({

        "Feature":
            X_sample.columns,

        "Mean_SHAP_Value":
            abs(values).mean(axis=0)

    })


    importance = importance.sort_values(
        "Mean_SHAP_Value",
        ascending=False
    )


    importance.to_csv(
        RESULTS_DIR /
        "shap_feature_importance.csv",
        index=False
    )


    print("\nTop SHAP Features")
    print(
        importance.head(10)
    )


    print(
        "✓ SHAP Analysis Complete"
    )


    return importance