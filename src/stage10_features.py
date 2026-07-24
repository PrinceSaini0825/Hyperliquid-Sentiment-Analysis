"""
Stage 10:
Advanced Trader Intelligence Features

Features:
- Trader performance history
- Rolling statistics
- Sentiment lag
- Market behavior
"""


import pandas as pd
import numpy as np



def add_trader_features(df):

    print("\nGenerating Trader Intelligence Features...")


    df = df.sort_values(
        [
            "account",
            "timestamp"
        ]
    )



    # ----------------------------
    # Trader Win Rate
    # ---------------------------



    # ----------------------------
    # Rolling PnL
    # ----------------------------


    df["rolling_pnl"] = (

        df
        .groupby("account")["closedpnl"]
        .transform(
            lambda x:
            x.shift()
             .rolling(
                 50,
                 min_periods=10
             )
             .sum()
        )

    )



    # ----------------------------
    # Average Trade Size
    # ----------------------------


    df["rolling_trade_size"] = (

        df
        .groupby("account")["sizeusd"]
        .transform(

            lambda x:
            x.shift()
             .rolling(
                 50,
                 min_periods=10
             )
             .mean()

        )

    )



    # ----------------------------
    # Profit Factor
    # ----------------------------




    # ----------------------------
    # Sentiment Lag
    # ----------------------------


    sentiment_map = {

        "Extreme Fear":0,

        "Fear":1,

        "Neutral":2,

        "Greed":3,

        "Extreme Greed":4

    }


    df["sentiment_score"] = (

        df["market_sentiment"]
        .map(sentiment_map)

    )



    df["sentiment_change"] = (

        df["sentiment_score"]
        .diff()

    )


    df["sentiment_rolling"] = (

        df["sentiment_score"]
        .rolling(
            5
        )
        .mean()

    )



    # ----------------------------
    # Fill Missing
    # ----------------------------


    df = df.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    df=df.fillna(0)



    print(
        "✓ Trader Features Added"
    )


    return df