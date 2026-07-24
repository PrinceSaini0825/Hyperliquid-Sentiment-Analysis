"""
Exploratory Data Analysis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULT_DIR = Path("results")
FIGURE_DIR = Path("figures")

RESULT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)


def save_plot(name):

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / name,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def generate_summary(df):

    summary = []

    summary.append(f"Rows : {len(df)}")
    summary.append(f"Columns : {len(df.columns)}")
    summary.append(f"Duplicate Rows : {df.duplicated().sum()}")

    summary.append("\nMissing Values")

    summary.append(str(df.isna().sum()))

    summary.append("\nStatistics")

    summary.append(str(df.describe(include="all")))

    with open(
        RESULT_DIR / "eda_summary.txt",
        "w"
    ) as f:

        f.write("\n".join(summary))


def sentiment_distribution(df):

    plt.figure(figsize=(8,5))

    df["market_sentiment"].value_counts().plot(
        kind="bar"
    )

    plt.title("Market Sentiment Distribution")

    plt.ylabel("Trades")

    save_plot("01_sentiment_distribution.png")


def pnl_distribution(df):

    plt.figure(figsize=(8,5))

    df["closedpnl"].hist(
        bins=60
    )

    plt.title("Closed PnL Distribution")

    plt.xlabel("PnL")

    save_plot("02_pnl_distribution.png")


def pnl_vs_sentiment(df):

    plt.figure(figsize=(8,5))

    df.boxplot(
        column="closedpnl",
        by="market_sentiment"
    )

    plt.title("PnL by Market Sentiment")

    plt.suptitle("")

    plt.xlabel("Sentiment")

    plt.ylabel("Closed PnL")

    save_plot("03_pnl_vs_sentiment.png")


def volume_vs_sentiment(df):

    volume = (
        df.groupby("market_sentiment")["sizeusd"]
        .sum()
    )

    plt.figure(figsize=(8,5))

    volume.plot(kind="bar")

    plt.title("Trading Volume by Sentiment")

    plt.ylabel("USD")

    save_plot("04_volume_vs_sentiment.png")


def top_coins(df):

    plt.figure(figsize=(10,6))

    df["coin"].value_counts().head(10).plot(
        kind="bar"
    )

    plt.title("Top 10 Traded Coins")

    save_plot("05_top_coins.png")


def daily_pnl(df):

    daily = (
        df.groupby("trade_date")["closedpnl"]
        .sum()
    )

    plt.figure(figsize=(14,5))

    daily.plot()

    plt.title("Daily Closed PnL")

    plt.ylabel("PnL")

    save_plot("06_daily_pnl.png")


def correlation(df):

    numeric = df.select_dtypes("number")

    corr = numeric.corr()

    plt.figure(figsize=(8,6))

    plt.imshow(corr)

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )

    plt.colorbar()

    plt.title("Correlation Matrix")

    save_plot("07_correlation_heatmap.png")


def run_eda(df):

    print("\nGenerating EDA...")

    generate_summary(df)

    sentiment_distribution(df)

    pnl_distribution(df)

    pnl_vs_sentiment(df)

    volume_vs_sentiment(df)

    top_coins(df)

    daily_pnl(df)

    correlation(df)

    print("EDA Complete.")