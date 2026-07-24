"""
Machine Learning Dataset Preparation

Features:
- Leakage-free dataset preparation
- Time based train-test split
- Categorical encoding
- Missing value handling
- Preprocessor saving
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


TARGET = "profit_flag"


# =====================================================
# Columns removed from ML
# =====================================================

DROP_COLUMNS = [
    # Identity
    "account",
    "transactionhash",

    # Raw time columns (avoid time leakage)
    "timestamp",
    "timestampist",
    "trade_date",

    # IDs
    "orderid",
    "tradeid",

    # Direct leakage
    "closedpnl",
    "profit_percentage",

    # Trade execution information unavailable before trade
    "direction",
    "side",
    "startposition"
    "previous_pnl",
"historical_avg_pnl",
"previous_win",
"historical_win_rate",
]

# =====================================================
# Prepare Features and Target
# =====================================================

def prepare_dataset(df):
    """
    Separate features and target.

    Removes leakage columns.
    """

    df = df.copy()


    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )


    y = df[TARGET]


    X = df.drop(
        columns=DROP_COLUMNS + [TARGET],
        errors="ignore"
    )


    return X, y



# =====================================================
# Encoding
# =====================================================

def fit_encode_dataframe(df):
    """
    Fit encoders only on training data.
    """

    df = df.copy()

    encoders = {}


    for col in df.select_dtypes(
        include="object"
    ).columns:


        encoder = LabelEncoder()


        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )


        encoders[col] = encoder


    return df, encoders



def transform_encode_dataframe(
        df,
        encoders
):
    """
    Transform test data using
    training encoders.
    """

    df = df.copy()


    for col, encoder in encoders.items():

        if col in df.columns:


            values = (
                df[col]
                .astype(str)
            )


            # Handle unseen categories

            values = values.apply(
                lambda x:
                x if x in encoder.classes_
                else encoder.classes_[0]
            )


            df[col] = encoder.transform(
                values
            )


    return df



# =====================================================
# Missing Value Handling
# =====================================================

def fit_imputer(df):
    """
    Fit imputer only on training data.
    """


    empty_cols = df.columns[
        df.isna().all()
    ].tolist()


    if empty_cols:

        print(
            "\nRemoving Empty Columns:"
        )

        print(empty_cols)


        df = df.drop(
            columns=empty_cols
        )


    imputer = SimpleImputer(
        strategy="median"
    )


    values = imputer.fit_transform(
        df
    )


    df = pd.DataFrame(
        values,
        columns=df.columns,
        index=df.index
    )


    return df, imputer



def transform_imputer(
        df,
        imputer,
        columns
):
    """
    Apply trained imputer.
    """


    values = imputer.transform(
        df
    )


    return pd.DataFrame(
        values,
        columns=columns,
        index=df.index
    )



# =====================================================
# Time Based Split
# =====================================================

def split_dataset(
        X,
        y
):
    """
    Preserve chronological order.
    """


    split_index = int(
        len(X) * 0.80
    )


    X_train = X.iloc[
        :split_index
    ]


    X_test = X.iloc[
        split_index:
    ]


    y_train = y.iloc[
        :split_index
    ]


    y_test = y.iloc[
        split_index:
    ]


    return (
        X_train,
        X_test,
        y_train,
        y_test
    )



# =====================================================
# Save preprocessors
# =====================================================

def save_preprocessors(
        encoders,
        imputer
):


    joblib.dump(
        encoders,
        MODEL_DIR /
        "label_encoders.pkl"
    )


    joblib.dump(
        imputer,
        MODEL_DIR /
        "imputer.pkl"
    )



# =====================================================
# Main ML Preparation
# =====================================================

def prepare_ml_data(df):

    print(
        "\nPreparing Machine Learning Dataset..."
    )


    # -----------------------------
    # Create X,y
    # -----------------------------

    X, y = prepare_dataset(
        df
    )


    # -----------------------------
    # Time split first
    # -----------------------------

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )


    # -----------------------------
    # Encoding
    # -----------------------------

    X_train, encoders = fit_encode_dataframe(
        X_train
    )


    X_test = transform_encode_dataframe(
        X_test,
        encoders
    )



    # -----------------------------
    # Imputation
    # -----------------------------

    X_train, imputer = fit_imputer(
        X_train
    )


    X_test = transform_imputer(
        X_test,
        imputer,
        X_train.columns
    )



    # -----------------------------
    # Save pipeline objects
    # -----------------------------

    save_preprocessors(
        encoders,
        imputer
    )



    print(
        "✓ ML Dataset Ready"
    )


    print(
        "\nDataset Summary"
    )

    print(
        "-" * 35
    )


    print(
        f"Training Samples : {len(X_train):,}"
    )


    print(
        f"Testing Samples  : {len(X_test):,}"
    )


    print(
        f"Number of Features : {X_train.shape[1]}"
    )



    return (

        X_train,
        X_test,
        y_train,
        y_test

    )