"""
Advanced Model Training Module

Stage 9:
- Model Optimization
- Threshold tuning
- Time series cross validation
- Best model selection
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)



# ============================================================
# Models
# ============================================================

MODELS = {


    "Logistic Regression":

    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
        random_state=42
    ),



    "Random Forest":

    RandomForestClassifier(

        n_estimators=250,

        max_depth=12,

        min_samples_split=10,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1

    ),



    "Gradient Boosting":

    GradientBoostingClassifier(

        n_estimators=150,

        learning_rate=0.03,

        max_depth=4,

        random_state=42

    )

}



# ============================================================
# Threshold Optimization
# ============================================================


def optimize_threshold(
        y_true,
        probabilities
):

    """
    Find best probability threshold using F1 score.
    """

    best_threshold = 0.5

    best_f1 = 0



    for threshold in np.arange(
            0.1,
            0.9,
            0.01
    ):


        predictions = (
            probabilities >= threshold
        ).astype(int)



        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )



        if score > best_f1:

            best_f1 = score

            best_threshold = threshold



    return (
        best_threshold,
        best_f1
    )





# ============================================================
# Evaluation
# ============================================================


def evaluate_model(
        model,
        X,
        y
):


    probability = model.predict_proba(
        X
    )[:,1]



    threshold, threshold_f1 = optimize_threshold(
        y,
        probability
    )



    prediction = (
        probability >= threshold
    ).astype(int)



    return {


        "Accuracy":

        accuracy_score(
            y,
            prediction
        ),



        "Precision":

        precision_score(
            y,
            prediction,
            zero_division=0
        ),



        "Recall":

        recall_score(
            y,
            prediction,
            zero_division=0
        ),



        "F1":

        f1_score(
            y,
            prediction,
            zero_division=0
        ),



        "ROC_AUC":

        roc_auc_score(
            y,
            probability
        ),



        "Threshold":

        threshold

    }





# ============================================================
# Cross Validation
# ============================================================


def cross_validate(
        model,
        X,
        y
):


    scores = []



    tscv = TimeSeriesSplit(
        n_splits=3
    )



    for train_idx, test_idx in tscv.split(X):


        X_train = X.iloc[train_idx]

        X_test = X.iloc[test_idx]


        y_train = y.iloc[train_idx]

        y_test = y.iloc[test_idx]



        model.fit(
            X_train,
            y_train
        )



        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )



        scores.append(
            metrics["F1"]
        )



    return np.mean(scores)





# ============================================================
# Training Pipeline
# ============================================================


def train_models(
        X_train,
        X_test,
        y_train,
        y_test
):


    print(
        "\nStage 9: Optimized Model Training..."
    )


    results = []



    best_model = None

    best_score = -1

    best_name = None



    for name, model in MODELS.items():



        print(
            f"\nTraining {name}"
        )



        # Train model

        model.fit(
            X_train,
            y_train
        )



        # Test metrics

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )



        # Time-series CV

        cv_score = cross_validate(
            model,
            X_train,
            y_train
        )



        metrics["CV_F1"] = cv_score

        metrics["Model"] = name



        results.append(
            metrics
        )



        # Select best model by CV F1

        if cv_score > best_score:


            best_score = cv_score

            best_model = model

            best_name = name





    results_df = pd.DataFrame(
        results
    )



    results_df = results_df[

        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "Threshold",
            "CV_F1"
        ]

    ]



    results_df.to_csv(

        RESULT_DIR /

        "stage9_model_results.csv",

        index=False

    )



    joblib.dump(

        best_model,

        MODEL_DIR /

        "stage9_best_model.pkl"

    )



    print(
        "\n==================="
    )


    print(
        "Best Model:",
        best_name
    )


    print(
        "CV F1:",
        round(best_score,4)
    )


    print(
        "==================="
    )



    print(
        results_df
    )



    return (

        best_model,

        results_df

    )





# ============================================================
# Feature Importance
# ============================================================


def save_feature_importance(
        model,
        X_train
):


    if hasattr(
            model,
            "feature_importances_"
    ):



        importance = pd.DataFrame({

            "Feature":
            X_train.columns,


            "Importance":
            model.feature_importances_

        })



        importance = importance.sort_values(

            "Importance",

            ascending=False

        )



        importance.to_csv(

            RESULT_DIR /

            "stage9_feature_importance.csv",

            index=False

        )



        print(
            "✓ Stage 9 Feature Importance Saved"
        )



    else:


        print(
            "Feature importance unavailable"
        )