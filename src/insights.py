"""
Automated Insights Engine
"""

from pathlib import Path
import pandas as pd

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def save_report(filename, text):

    with open(
        REPORT_DIR / filename,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(text)


def executive_summary(df):

    total = len(df)

    matched = df["market_sentiment"].notna().sum()

    coverage = matched / total * 100

    avg_pnl = df["closedpnl"].mean()

    total_profit = df["closedpnl"].sum()

    profitable = (
        df["closedpnl"] > 0
    ).sum()

    losing = (
        df["closedpnl"] <= 0
    ).sum()

    report = f"""
# Executive Summary

## Dataset Overview

Total Trades : {total:,}

Matched With Sentiment : {matched:,}

Coverage : {coverage:.2f}%

Average Closed PnL : {avg_pnl:.2f}

Total Closed PnL : {total_profit:.2f}

Profitable Trades : {profitable:,}

Loss Trades : {losing:,}

"""

    save_report(
        "executive_summary.md",
        report
    )


def sentiment_statistics(df):

    report = "# Sentiment Analysis\n\n"

    stats = (

        df.groupby(
            "market_sentiment"
        )["closedpnl"]

        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max"
            ]
        )

    )

    report += stats.to_markdown()

    save_report(
        "market_analysis.md",
        report
    )
def trader_statistics(df):

    report = "# Trader Behaviour\n\n"

    trader = (

        df.groupby(
            "account"
        )["closedpnl"]

        .agg(
            [
                "count",
                "sum",
                "mean",
                "median"
            ]
        )

        .sort_values(
            "sum",
            ascending=False
        )

    )

    report += trader.head(20).to_markdown()

    save_report(
        "trader_behavior.md",
        report
    )


def top_coin_analysis(df):

    report = "# Coin Analysis\n\n"

    coins = (

        df.groupby(
            "coin"
        )["closedpnl"]

        .sum()

        .sort_values(
            ascending=False
        )

    )

    report += "## Top Performing Coins\n\n"

    report += coins.head(15).to_markdown()

    report += "\n\n"

    report += "## Worst Performing Coins\n\n"

    report += coins.tail(15).to_markdown()

    save_report(
        "insights.md",
        report
    )
def recommendations(df):

    report = "# Trading Recommendations\n\n"

    pnl = (
        df.groupby(
            "market_sentiment"
        )["closedpnl"]
        .mean()
    )

    best = pnl.idxmax()

    worst = pnl.idxmin()

    report += f"""

Highest Average PnL occurred during:

**{best}**

Lowest Average PnL occurred during:

**{worst}**

Recommendations

• Increase exposure during historically profitable sentiment.

• Reduce position size during historically weak sentiment.

• Monitor transaction fees before increasing trade frequency.

• Diversify across high-performing assets.

• Use market sentiment together with technical indicators.

"""

    save_report(
        "recommendations.md",
        report
    )


def generate_insights(df):

    print("\nGenerating Automated Reports...")

    executive_summary(df)

    sentiment_statistics(df)

    trader_statistics(df)

    top_coin_analysis(df)

    recommendations(df)

    print("✓ Reports Generated")