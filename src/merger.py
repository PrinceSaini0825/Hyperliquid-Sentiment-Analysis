"""
Production Smart Merge Engine
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.loader import COLUMN_MAP, find_column
from src.utils import log

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)


def prepare_trade_date(df):

    time_col = find_column(df, COLUMN_MAP["time"])

    if time_col is None:
        raise ValueError("Historical timestamp column not found.")

    df["trade_date"] = pd.to_datetime(df[time_col]).dt.date

    return df


def prepare_sentiment(df):

    if "date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["date"]).dt.date

    else:

        time_col = find_column(df, COLUMN_MAP["time"])

        if time_col is None:
            raise ValueError("Sentiment date column not found.")

        df["trade_date"] = pd.to_datetime(df[time_col]).dt.date

    sentiment_col = find_column(df, COLUMN_MAP["sentiment"])

    if sentiment_col is None:
        raise ValueError("Sentiment column not found.")

    value_col = None

    if "value" in df.columns:
        value_col = "value"

    return df, sentiment_col, value_col


def merge_data(historical, fear):

    historical = prepare_trade_date(historical)

    fear, sentiment_col, value_col = prepare_sentiment(fear)

    fear = fear.sort_values("trade_date")

    fear = fear.drop_duplicates(
        subset="trade_date",
        keep="last"
    )

    columns = ["trade_date", sentiment_col]

    if value_col:
        columns.append(value_col)

    merged = historical.merge(
        fear[columns],
        on="trade_date",
        how="left"
    )

    rename = {
        sentiment_col: "market_sentiment"
    }

    if value_col:
        rename[value_col] = "fear_greed_value"

    merged.rename(columns=rename, inplace=True)

    matched = merged["market_sentiment"].notna().sum()

    unmatched = merged["market_sentiment"].isna().sum()

    coverage = matched / len(merged) * 100

    print("\nMerge Quality Report")

    print("=" * 60)

    print(f"Total Trades      : {len(merged):,}")

    print(f"Matched Trades    : {matched:,}")

    print(f"Unmatched Trades  : {unmatched:,}")

    print(f"Coverage          : {coverage:.2f}%")

    report = pd.DataFrame({

        "Metric":[
            "Total Trades",
            "Matched",
            "Unmatched",
            "Coverage (%)"
        ],

        "Value":[
            len(merged),
            matched,
            unmatched,
            round(coverage,2)
        ]

    })

    report.to_csv(
        RESULT_DIR / "merge_report.csv",
        index=False
    )

    merged.to_csv(
        RESULT_DIR / "merged.csv",
        index=False
    )

    summary = {

        "total_trades": int(len(merged)),
        "matched": int(matched),
        "unmatched": int(unmatched),
        "coverage": round(float(coverage),2)

    }

    with open(
        RESULT_DIR / "merge_summary.json",
        "w"
    ) as f:

        json.dump(summary, f, indent=4)

    log("Merge completed successfully.")

    return merged