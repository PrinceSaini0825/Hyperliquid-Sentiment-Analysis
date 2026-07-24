"""
Data Preprocessing Module
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.loader import COLUMN_MAP, find_column
from src.utils import log

# Ensure results directory exists
Path("results").mkdir(exist_ok=True)

# Logical numeric columns
NUMERIC_COLUMNS = [
    "price",
    "size",
    "pnl",
    "leverage",
]


def parse_datetime(series: pd.Series) -> pd.Series:
    """
    Automatically parse datetime columns.

    Supports:
    - Unix timestamps (seconds)
    - Unix timestamps (milliseconds)
    - ISO datetime strings
    """

    if pd.api.types.is_numeric_dtype(series):

        clean = series.dropna()

        if len(clean) > 0:

            median = clean.median()

            if median > 1e12:
                return pd.to_datetime(series, unit="ms", errors="coerce")

            elif median > 1e9:
                return pd.to_datetime(series, unit="s", errors="coerce")

    return pd.to_datetime(series, errors="coerce")


def convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert important columns to numeric.
    """

    for logical_name in NUMERIC_COLUMNS:

        column = find_column(df, COLUMN_MAP.get(logical_name, []))

        if column:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def preprocess_dataframe(
    df: pd.DataFrame,
    dataset_name: str
) -> pd.DataFrame:
    """
    Clean a dataframe.
    """

    print(f"\nCleaning {dataset_name}...")

    before = len(df)

    # Remove duplicates
    df = df.drop_duplicates().copy()

    removed = before - len(df)

    log(f"{dataset_name}: Removed {removed} duplicate rows")

    # Parse datetime
    time_col = find_column(df, COLUMN_MAP["time"])

    if time_col:

        df[time_col] = parse_datetime(df[time_col])

    # Convert numerics
    df = convert_numeric(df)

    # Fill missing numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:

        median = df[col].median()

        df[col] = df[col].fillna(median)

    # Fill missing text values
    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:

        df[col] = df[col].fillna("Unknown")

    print("Done.")

    return df


def preprocess_data(
    historical: pd.DataFrame,
    fear: pd.DataFrame
):
    """
    Preprocess both datasets.
    """

    historical = preprocess_dataframe(
        historical,
        "Historical Dataset"
    )

    fear = preprocess_dataframe(
        fear,
        "Fear & Greed Dataset"
    )

    # Save cleaned datasets
    historical.to_csv(
        "results/historical_clean.csv",
        index=False
    )

    fear.to_csv(
        "results/fear_greed_clean.csv",
        index=False
    )

    log("Cleaned datasets saved.")

    print("\n✓ Cleaned datasets saved in results/")

    return historical, fear