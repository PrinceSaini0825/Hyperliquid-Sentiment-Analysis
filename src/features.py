"""
Feature Engineering Module

Leakage-safe feature creation for
Hyperliquid Sentiment Analysis
"""


from pathlib import Path

import numpy as np
import pandas as pd


RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)



# -------------------------------------------------
# Time Features
# -------------------------------------------------

def create_time_features(df):

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )


    df["trade_date"] = (
        df["timestamp"]
        .dt.date
    )

    df["trade_year"] = (
        df["timestamp"]
        .dt.year
    )

    df["trade_month"] = (
        df["timestamp"]
        .dt.month
    )

    df["trade_day"] = (
        df["timestamp"]
        .dt.day
    )

    df["hour"] = (
        df["timestamp"]
        .dt.hour
    )

    df["day_of_week"] = (
        df["timestamp"]
        .dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)


    return df




# -------------------------------------------------
# Target Creation
# -------------------------------------------------

def create_profit_features(df):

    df["profit_flag"] = (
        df["closedpnl"] > 0
    ).astype(int)

    return df




# -------------------------------------------------
# Trade Size Features
# -------------------------------------------------

def create_size_features(df):

    df["log_trade_size"] = (
        np.log1p(
            abs(df["sizeusd"])
        )
    )


    return df




# -------------------------------------------------
# Sentiment Encoding
# -------------------------------------------------

def create_sentiment_features(df):

    mapping = {

        "Extreme Fear":0,

        "Fear":1,

        "Neutral":2,

        "Greed":3,

        "Extreme Greed":4

    }


    df["sentiment_encoded"] = (
        df["market_sentiment"]
        .map(mapping)
        .fillna(-1)
    )


    return df




# -------------------------------------------------
# Trader Behaviour Features
# -------------------------------------------------

def create_trader_features(df):
    """
    Trader behavior features without using PnL.
    Prevent target leakage.
    """

    df = df.sort_values(
        ["account", "timestamp"]
    )


    # Number of previous trades
    df["trader_trade_count"] = (
        df.groupby("account")
        .cumcount()
    )


    # Previous trading activity size
    df["previous_trade_size"] = (
        df.groupby("account")
        ["sizeusd"]
        .shift(1)
    )


    # Average previous trade size
    df["historical_avg_trade_size"] = (
        df.groupby("account")
        ["sizeusd"]
        .transform(
            lambda x:
            x.shift(1)
             .expanding()
             .mean()
        )
    )


    # Trading frequency proxy
    df["trader_activity_ratio"] = (
        df["sizeusd"] /
        (df["historical_avg_trade_size"] + 1)
    )


    df[
        [
            "previous_trade_size",
            "historical_avg_trade_size",
            "trader_activity_ratio"
        ]
    ] = df[
        [
            "previous_trade_size",
            "historical_avg_trade_size",
            "trader_activity_ratio"
        ]
    ].fillna(0)


    return df

# -------------------------------------------------
# Coin Activity Features
# -------------------------------------------------

def create_coin_features(df):


    coin_frequency = (
        df["coin"]
        .value_counts()
    )


    df["coin_frequency"] = (
        df["coin"]
        .map(coin_frequency)
    )


    return df


# -------------------------------------------------
# Market Microstructure Features
# -------------------------------------------------

def create_market_features(df):

    df = df.sort_values(
        "timestamp"
    )


    # Price movement proxy
    df["price_change"] = (
        df["executionprice"]
        .pct_change()
        .fillna(0)
    )


    # Current trade size compared with recent activity

    rolling_size = (
        df["sizeusd"]
        .rolling(
            100,
            min_periods=1
        )
        .mean()
    )


    df["trade_size_ratio"] = (
        df["sizeusd"]
        /
        (rolling_size + 1e-9)
    )


    # Fee pressure

    df["fee_ratio"] = (
        df["fee"]
        /
        (abs(df["sizeusd"]) + 1e-9)
    )


    return df

# -------------------------------------------------
# Market Activity Features
# -------------------------------------------------

def create_rolling_features(df):

    daily_volume = (

        df.groupby(
            "trade_date"
        )["sizeusd"]
        .sum()
        .sort_index()

    )


    rolling7 = (
        daily_volume
        .rolling(
            7,
            min_periods=1
        )
        .mean()
    )


    rolling30 = (
        daily_volume
        .rolling(
            30,
            min_periods=1
        )
        .mean()
    )


    df["rolling7_volume"] = (
        df["trade_date"]
        .map(rolling7)
    )


    df["rolling30_volume"] = (
        df["trade_date"]
        .map(rolling30)
    )


    return df




# -------------------------------------------------
# Save
# -------------------------------------------------

def save_features(df):


    df.to_csv(

        RESULT_DIR /
        "feature_engineered.csv",

        index=False

    )


    summary = pd.DataFrame({

        "Feature":
        df.columns,


        "Type":
        df.dtypes.astype(str)

    })


    summary.to_csv(

        RESULT_DIR /
        "feature_summary.csv",

        index=False

    )




# -------------------------------------------------
# Main Pipeline
# -------------------------------------------------

def engineer_features(df):


    print("\nEngineering Features...")


    df = create_time_features(df)

    df = create_profit_features(df)

    df = create_size_features(df)

    df = create_sentiment_features(df)

    df = create_trader_features(df)

    df = create_coin_features(df)

    df = create_market_features(df)

    df = create_rolling_features(df)


    save_features(df)


    print("✓ Feature Engineering Complete")


    return df