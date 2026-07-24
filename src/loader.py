"""
Universal Data Loader
"""

from pathlib import Path
import re

import chardet
import pandas as pd

from config import FEAR_GREED_DATA, HISTORICAL_DATA
from src.utils import log


COLUMN_MAP = {
    "time": ["time", "timestamp", "datetime", "date"],
    "symbol": ["symbol", "coin", "asset"],
    "price": ["executionprice", "price", "execution_price"],
    "size": ["size", "quantity", "sizeusd"],
    "side": ["side", "direction"],
    "pnl": ["closedpnl", "closed_pnl", "realizedpnl", "realized_pnl", "pnl"],
    "leverage": ["leverage"],
    "sentiment": ["classification", "sentiment"],
}


def normalize_column(name: str) -> str:
    """
    Standardize a column name.
    """
    name = name.lower().strip()
    name = re.sub(r"[\s_\-/()]+", "", name)
    return name


def detect_encoding(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        raw = f.read()

    encoding = chardet.detect(raw)["encoding"]

    log(f"Detected encoding {encoding}")

    return encoding


def load_csv(file_path: Path) -> pd.DataFrame:

    encoding = detect_encoding(file_path)

    df = pd.read_csv(file_path, encoding=encoding)

    original_columns = list(df.columns)

    df.columns = [normalize_column(c) for c in df.columns]

    log(f"Loaded {file_path.name}")

    print(f"\nLoaded: {file_path.name}")

    print("\nOriginal Columns")

    print(original_columns)

    print("\nNormalized Columns")

    print(df.columns.tolist())

    return df


def find_column(df: pd.DataFrame, aliases: list[str]):

    for alias in aliases:
        if alias in df.columns:
            return alias

    return None


def validate_dataframe(df: pd.DataFrame, name: str):

    print("\n" + "=" * 60)

    print(name)

    print("=" * 60)

    print(f"Rows        : {len(df):,}")

    print(f"Columns     : {len(df.columns)}")

    print(f"Duplicates  : {df.duplicated().sum()}")

    print(f"Missing     : {df.isna().sum().sum()}")

    print("\nDetected Important Columns")

    for key, aliases in COLUMN_MAP.items():

        column = find_column(df, aliases)

        print(f"{key:12} -> {column}")


def load_data():

    historical = load_csv(HISTORICAL_DATA)

    fear = load_csv(FEAR_GREED_DATA)

    validate_dataframe(historical, "Historical Dataset")

    validate_dataframe(fear, "Fear & Greed Dataset")

    return historical, fear