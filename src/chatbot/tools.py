"""
Chatbot Analytics Tools
"""

from functools import lru_cache
from contextlib import redirect_stdout
from pathlib import Path

import io
import pandas as pd
import numpy as np

from src.loader import load_data
from src.preprocess import preprocess_data


@lru_cache(maxsize=1)
def load_project_data():
    """
    Load and preprocess project datasets only once.
    Suppress console output from loader/preprocessor.
    """

    with redirect_stdout(io.StringIO()):

        historical, fear = load_data()

        historical, fear = preprocess_data(
            historical,
            fear
        )

    return historical, fear



def dataset_summary():

    historical, fear = load_project_data()

    return {
        "historical_rows": len(historical),
        "historical_columns": list(historical.columns),
        "fear_rows": len(fear),
        "fear_columns": list(fear.columns)
    }



def pnl_statistics():

    historical, _ = load_project_data()

    pnl = historical["closedpnl"]

    return {

        "mean": float(pnl.mean()),

        "median": float(pnl.median()),

        "max_profit": float(pnl.max()),

        "max_loss": float(pnl.min()),

        "total_profit": float(pnl.sum())

    }



def top_coins(n=10):

    historical, _ = load_project_data()

    return (
        historical["coin"]
        .value_counts()
        .head(n)
        .to_dict()
    )



def buy_sell_ratio():

    historical, _ = load_project_data()

    return (
        historical["side"]
        .value_counts()
        .to_dict()
    )



def sentiment_distribution():

    _, fear = load_project_data()

    return (
        fear["classification"]
        .value_counts()
        .to_dict()
    )



def latest_predictions():

    file = Path(
        "results/predictions.csv"
    )

    if not file.exists():

        return "Run prediction pipeline first."


    return (
        pd.read_csv(file)
        .head(20)
        .to_dict("records")
    )



def latest_signals():

    file = Path(
        "results/signals/signals.csv"
    )

    if not file.exists():

        return "Run Stage 15 first."


    df = (
        pd.read_csv(file)
        .tail(20)
    )


    summary = {

        "Latest Asset": df["Coin"].iloc[-1],

        "Total Signals": len(df),


        "BUY Signals": int(
            (df["Signal"] == "BUY")
            .sum()
        ),


        "SELL Signals": int(
            (df["Signal"] == "SELL")
            .sum()
        ),


        "HOLD Signals": int(
            (df["Signal"] == "HOLD")
            .sum()
        ),


        "Average Confidence": float(
    round(
        df["Confidence"].mean(),
        3
    )
),


        "Risk Level": (
            df["Risk"]
            .mode()[0]
        )

    }


    return summary