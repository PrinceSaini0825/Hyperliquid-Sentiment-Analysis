"""
Professional Visualization Module
Hyperliquid Sentiment Analysis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Helper
# --------------------------------------------------

def save_plot(filename):

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# 1 Sentiment Distribution
# --------------------------------------------------

def sentiment_distribution(df):

    if "market_sentiment" not in df.columns:
        return

    plt.figure(figsize=(8,5))

    (
        df["market_sentiment"]
        .fillna("Unknown")
        .value_counts()
        .plot(kind="bar")
    )

    plt.title("Market Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Trades")

    save_plot("01_sentiment_distribution.png")


# --------------------------------------------------
# 2 PnL Distribution
# --------------------------------------------------

def pnl_distribution(df):

    if "closedpnl" not in df.columns:
        return

    plt.figure(figsize=(8,5))

    df["closedpnl"].hist(
        bins=80
    )

    plt.title("PnL Distribution")
    plt.xlabel("Closed PnL")
    plt.ylabel("Frequency")

    save_plot("02_pnl_distribution.png")


# --------------------------------------------------
# 3 Boxplot
# --------------------------------------------------

def pnl_boxplot(df):

    if (
        "closedpnl" not in df.columns
        or
        "market_sentiment" not in df.columns
    ):
        return

    plt.figure(figsize=(8,6))

    df.boxplot(
        column="closedpnl",
        by="market_sentiment"
    )

    plt.title("PnL by Sentiment")
    plt.suptitle("")
    plt.xlabel("Sentiment")
    plt.ylabel("Closed PnL")

    save_plot("03_boxplot_sentiment.png")


# --------------------------------------------------
# 4 Volume by Sentiment
# --------------------------------------------------

def volume_by_sentiment(df):

    if (
        "market_sentiment" not in df.columns
        or
        "sizeusd" not in df.columns
    ):
        return

    volume = (

        df.groupby(
            "market_sentiment"
        )["sizeusd"]

        .sum()

    )

    plt.figure(figsize=(8,5))

    volume.plot(kind="bar")

    plt.title("Trading Volume by Sentiment")
    plt.ylabel("USD")

    save_plot("04_volume_sentiment.png")


# --------------------------------------------------
# 5 Top Coins
# --------------------------------------------------

def top_coins(df):

    if "coin" not in df.columns:
        return

    plt.figure(figsize=(10,5))

    (

        df["coin"]

        .value_counts()

        .head(10)

        .plot(kind="bar")

    )

    plt.title("Top 10 Coins")

    save_plot("05_top_coins.png")


# --------------------------------------------------
# 6 Daily PnL
# --------------------------------------------------

def daily_pnl(df):

    if (
        "trade_date" not in df.columns
        or
        "closedpnl" not in df.columns
    ):
        return

    daily = (

        df.groupby(
            "trade_date"
        )["closedpnl"]

        .sum()

    )

    plt.figure(figsize=(12,5))

    daily.plot()

    plt.title("Daily Closed PnL")

    plt.ylabel("PnL")

    save_plot("06_daily_pnl.png")


# --------------------------------------------------
# 7 Cumulative PnL
# --------------------------------------------------

def cumulative_pnl(df):

    if (
        "trade_date" not in df.columns
        or
        "closedpnl" not in df.columns
    ):
        return

    pnl = (

        df.groupby(
            "trade_date"
        )["closedpnl"]

        .sum()

        .cumsum()

    )

    plt.figure(figsize=(12,5))

    pnl.plot()

    plt.title("Cumulative PnL")

    plt.ylabel("PnL")

    save_plot("07_cumulative_pnl.png")


# --------------------------------------------------
# 8 Daily Trade Count
# --------------------------------------------------

def daily_trade_count(df):

    if "trade_date" not in df.columns:
        return

    daily = (

        df.groupby(
            "trade_date"
        )

        .size()

    )

    plt.figure(figsize=(12,5))

    daily.plot()

    plt.title("Daily Trade Count")

    plt.ylabel("Trades")

    save_plot("08_trade_count_daily.png")
    # --------------------------------------------------
# 9 Monthly Trade Count
# --------------------------------------------------

def monthly_trade_count(df):

    if "trade_date" not in df.columns:
        return

    tmp = df.copy()

    tmp["month"] = pd.to_datetime(
        tmp["trade_date"]
    ).dt.to_period("M").astype(str)

    monthly = tmp.groupby("month").size()

    plt.figure(figsize=(12,5))

    monthly.plot(kind="bar")

    plt.title("Monthly Trade Count")
    plt.xlabel("Month")
    plt.ylabel("Trades")

    save_plot("09_monthly_trade_count.png")


# --------------------------------------------------
# 10 Monthly PnL
# --------------------------------------------------

def monthly_pnl(df):

    if (
        "trade_date" not in df.columns
        or
        "closedpnl" not in df.columns
    ):
        return

    tmp = df.copy()

    tmp["month"] = pd.to_datetime(
        tmp["trade_date"]
    ).dt.to_period("M").astype(str)

    monthly = tmp.groupby("month")["closedpnl"].sum()

    plt.figure(figsize=(12,5))

    monthly.plot(kind="bar")

    plt.title("Monthly Closed PnL")
    plt.xlabel("Month")
    plt.ylabel("PnL")

    save_plot("10_monthly_pnl.png")


# --------------------------------------------------
# 11 Buy Sell Ratio
# --------------------------------------------------

def buy_sell_ratio(df):

    if "side" not in df.columns:
        return

    plt.figure(figsize=(6,6))

    df["side"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%"
    )

    plt.ylabel("")
    plt.title("Buy vs Sell Ratio")

    save_plot("11_buy_sell_ratio.png")


# --------------------------------------------------
# 12 Top Profitable Coins
# --------------------------------------------------

def profitable_coins(df):

    if (
        "coin" not in df.columns
        or
        "closedpnl" not in df.columns
    ):
        return

    top = (

        df.groupby("coin")["closedpnl"]

        .sum()

        .sort_values(ascending=False)

        .head(10)

    )

    plt.figure(figsize=(10,5))

    top.plot(kind="bar")

    plt.title("Top 10 Profitable Coins")
    plt.ylabel("PnL")

    save_plot("12_top10_profitable.png")


# --------------------------------------------------
# 13 Top Losing Coins
# --------------------------------------------------

def losing_coins(df):

    if (
        "coin" not in df.columns
        or
        "closedpnl" not in df.columns
    ):
        return

    low = (

        df.groupby("coin")["closedpnl"]

        .sum()

        .sort_values()

        .head(10)

    )

    plt.figure(figsize=(10,5))

    low.plot(kind="bar")

    plt.title("Top 10 Losing Coins")
    plt.ylabel("PnL")

    save_plot("13_top10_loss.png")


# --------------------------------------------------
# 14 Trade Size Distribution
# --------------------------------------------------

def trade_size_distribution(df):

    if "sizeusd" not in df.columns:
        return

    plt.figure(figsize=(8,5))

    df["sizeusd"].hist(
        bins=80
    )

    plt.title("Trade Size Distribution")
    plt.xlabel("Trade Size (USD)")
    plt.ylabel("Frequency")

    save_plot("14_trade_size_distribution.png")


# --------------------------------------------------
# 15 Rolling 7-Day PnL
# --------------------------------------------------

def rolling_pnl(df):

    if (
        "trade_date" not in df.columns
        or
        "closedpnl" not in df.columns
    ):
        return

    rolling = (

        df.groupby("trade_date")["closedpnl"]

        .sum()

        .rolling(7)

        .mean()

    )

    plt.figure(figsize=(12,5))

    rolling.plot()

    plt.title("7-Day Rolling Average PnL")
    plt.ylabel("PnL")

    save_plot("15_rolling_pnl.png")


# --------------------------------------------------
# 16 Sentiment Timeline
# --------------------------------------------------

def sentiment_timeline(df):

    if (
        "trade_date" not in df.columns
        or
        "market_sentiment" not in df.columns
    ):
        return

    timeline = (

        df.groupby(
            [
                "trade_date",
                "market_sentiment"
            ]
        )

        .size()

        .unstack(fill_value=0)

    )

    plt.figure(figsize=(14,5))

    timeline.plot(ax=plt.gca())

    plt.title("Sentiment Timeline")
    plt.ylabel("Trades")

    save_plot("16_sentiment_timeline.png")


# --------------------------------------------------
# 17 Correlation Heatmap
# --------------------------------------------------

def correlation_heatmap(df):

    cols = []

    for col in [
        "executionprice",
        "sizeusd",
        "closedpnl",
        "fee",
        "startposition"
    ]:
        if col in df.columns:
            cols.append(col)

    if len(cols) < 2:
        return

    corr = df[cols].corr(numeric_only=True)

    plt.figure(figsize=(8,6))

    plt.imshow(
        corr,
        aspect="auto"
    )

    plt.xticks(
        range(len(cols)),
        cols,
        rotation=45
    )

    plt.yticks(
        range(len(cols)),
        cols
    )

    plt.colorbar()

    plt.title("Correlation Heatmap")

    save_plot("17_correlation_heatmap.png")


# --------------------------------------------------
# Run Everything
# --------------------------------------------------

def run_visualizations(df):

    print("\nGenerating Professional Charts...")

    sentiment_distribution(df)

    pnl_distribution(df)

    pnl_boxplot(df)

    volume_by_sentiment(df)

    top_coins(df)

    daily_pnl(df)

    cumulative_pnl(df)

    daily_trade_count(df)

    monthly_trade_count(df)

    monthly_pnl(df)

    buy_sell_ratio(df)

    profitable_coins(df)

    losing_coins(df)

    trade_size_distribution(df)

    rolling_pnl(df)

    sentiment_timeline(df)

    correlation_heatmap(df)

    print("✓ All Visualization Charts Generated")