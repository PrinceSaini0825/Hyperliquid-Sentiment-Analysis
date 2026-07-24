"""
Stage 9:
Advanced Model Optimization

Models:
- XGBoost
- LightGBM
- CatBoost

Features:
- Class imbalance handling
- Threshold optimization
- Model comparison
- Feature importance
- Best model saving
"""

from pathlib import Path

import joblib
import pandas as pd
import numpy as np


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


RESULT_DIR = Path("results")
MODEL_DIR = Path("models")

RESULT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)



# ------------------------------------------------
# Threshold Optimization
# ------------------------------------------------

def find_best_threshold(
        y_true,
        probabilities
):

    thresholds = np.arange(
        0.05,
        0.95,
        0.01
    )

    best_threshold = 0.5
    best_f1 = 0


    for t in thresholds:

        pred = (
            probabilities >= t
        ).astype(int)


        score = f1_score(
            y_true,
            pred
        )


        if score > best_f1:

            best_f1 = score
            best_threshold = t


    return best_threshold, best_f1




# ------------------------------------------------
# Evaluation
# ------------------------------------------------

def evaluate_model(
        name,
        model,
        X_test,
        y_test
):


    prob = model.predict_proba(
        X_test
    )[:,1]


    threshold, best_f1 = find_best_threshold(
        y_test,
        prob
    )


    pred = (
        prob >= threshold
    ).astype(int)



    result = {


        "Model": name,


        "Threshold":
            threshold,


        "Accuracy":
            accuracy_score(
                y_test,
                pred
            ),


        "Precision":
            precision_score(
                y_test,
                pred
            ),


        "Recall":
            recall_score(
                y_test,
                pred
            ),


        "F1":
            best_f1,


        "ROC_AUC":
            roc_auc_score(
                y_test,
                prob
            )

    }


    return result, threshold




# ------------------------------------------------
# Training
# ------------------------------------------------


def train_stage9(
        X_train,
        X_test,
        y_train,
        y_test
):


    print("\n================================")
    print("Stage 9 Advanced Optimization")
    print("================================")


    results=[]

    models={}



    # ==============================
    # XGBoost
    # ==============================


    from xgboost import XGBClassifier



    scale_weight = (
        len(y_train[y_train==0]) /
        len(y_train[y_train==1])
    )



    xgb = XGBClassifier(

        n_estimators=500,

        learning_rate=0.03,

        max_depth=5,

        subsample=0.8,

        colsample_bytree=0.8,


        scale_pos_weight=
        scale_weight,


        random_state=42,

        eval_metric="logloss"

    )


    print("\nTraining XGBoost...")


    xgb.fit(
        X_train,
        y_train
    )


    models["XGBoost"]=xgb



    result,threshold=evaluate_model(
        "XGBoost",
        xgb,
        X_test,
        y_test
    )


    results.append(result)





    # ==============================
    # LightGBM
    # ==============================


    try:

        from lightgbm import LGBMClassifier



        lgbm=LGBMClassifier(

            n_estimators=500,

            learning_rate=0.03,

            num_leaves=32,

            max_depth=6,


            class_weight="balanced",

            random_state=42

        )


        print(
            "\nTraining LightGBM..."
        )


        lgbm.fit(
            X_train,
            y_train
        )


        models["LightGBM"]=lgbm



        result,_=evaluate_model(
            "LightGBM",
            lgbm,
            X_test,
            y_test
        )


        results.append(result)



    except Exception as e:

        print(
            "LightGBM skipped:",
            e
        )





    # ==============================
    # CatBoost
    # ==============================


    try:


        from catboost import CatBoostClassifier



        cat = CatBoostClassifier(

            iterations=500,

            learning_rate=0.03,

            depth=6,


            loss_function="Logloss",


            verbose=0,


            auto_class_weights="Balanced",

            random_seed=42

        )


        print(
            "\nTraining CatBoost..."
        )


        cat.fit(
            X_train,
            y_train
        )



        models["CatBoost"]=cat



        result,_=evaluate_model(

            "CatBoost",

            cat,

            X_test,

            y_test

        )


        results.append(result)



    except Exception as e:


        print(
            "CatBoost skipped:",
            e
        )





    # ==============================
    # Results
    # ==============================


    result_df=pd.DataFrame(
        results
    )


    result_df=result_df.sort_values(
        "ROC_AUC",
        ascending=False
    )



    print(
        "\nFinal Stage 9 Results"
    )


    print(
        result_df
    )



    result_df.to_csv(

        RESULT_DIR /
        "stage9_model_results.csv",

        index=False

    )




    # ==============================
    # Save Best Model
    # ==============================


    best_name = (
        result_df.iloc[0]["Model"]
    )


    best_model=models[
        best_name
    ]


    joblib.dump(

        best_model,

        MODEL_DIR /
        "best_model.pkl"

    )


    print(
        "\nBest Model Saved:",
        best_name
    )


    return result_df