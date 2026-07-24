"""
Prediction Module
"""

from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def load_model():
    """
    Load trained model.
    """

    model = joblib.load(
        MODEL_DIR / "best_model.pkl"
    )

    encoders = joblib.load(
        MODEL_DIR / "label_encoders.pkl"
    )

    imputer = joblib.load(
        MODEL_DIR / "imputer.pkl"
    )

    return model, encoders, imputer


def encode(df, encoders):

    df = df.copy()

    for column, encoder in encoders.items():

        if column in df.columns:

            values = df[column].astype(str)

            known = set(encoder.classes_)

            values = values.apply(
                lambda x: x if x in known else encoder.classes_[0]
            )

            df[column] = encoder.transform(values)

    return df


def preprocess(df, encoders, imputer):

    df = encode(df, encoders)

    values = imputer.transform(df)

    df = pd.DataFrame(
        values,
        columns=df.columns,
        index=df.index
    )

    return df


def predict(df):

    print("\nGenerating Predictions...")

    model, encoders, imputer = load_model()

    X = preprocess(
        df,
        encoders,
        imputer
    )

    predictions = model.predict(X)

    probability = model.predict_proba(X)[:, 1]

    output = df.copy()

    output["prediction"] = predictions

    output["confidence"] = probability

    output.to_csv(
        RESULT_DIR / "predictions.csv",
        index=False
    )

    print("✓ Predictions Saved")

    return output