"""
Professional Statistical Analysis
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPORT_DIR = Path("reports")
RESULT_DIR = Path("results")

REPORT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def confidence_interval(series):

    series = series.dropna()

    if len(series) < 2:
        return np.nan, np.nan

    mean = series.mean()

    ci = stats.t.interval(
        0.95,
        len(series)-1,
        loc=mean,
        scale=stats.sem(series)
    )

    return ci


def cohen_d(x, y):

    x = x.dropna()
    y = y.dropna()

    nx = len(x)
    ny = len(y)

    pooled = np.sqrt(
        (
            ((nx-1)*x.var()) +
            ((ny-1)*y.var())
        ) / (nx+ny-2)
    )

    return (x.mean()-y.mean()) / pooled


def descriptive_statistics(df):

    pnl = df["closedpnl"]

    return {

        "Trades": len(df),

        "Mean PnL": pnl.mean(),

        "Median PnL": pnl.median(),

        "Std": pnl.std(),

        "Minimum": pnl.min(),

        "Maximum": pnl.max(),

        "Win Rate %":
            (pnl > 0).mean()*100,

        "Loss Rate %":
            (pnl < 0).mean()*100,

        "Average Trade Size":
            df["sizeusd"].mean()

    }


def sentiment_statistics(df):

    summary = (
        df.groupby("market_sentiment")
        .agg(

            Trades=("closedpnl","count"),

            MeanPnL=("closedpnl","mean"),

            MedianPnL=("closedpnl","median"),

            StdPnL=("closedpnl","std"),

            WinRate=("closedpnl",
                     lambda x:(x>0).mean()*100),

            AvgVolume=("sizeusd","mean")

        )

    )

    summary.to_csv(
        RESULT_DIR/"statistics.csv"
    )

    return summary


def t_test(df):

    fear = df[
        df["market_sentiment"]=="Fear"
    ]["closedpnl"]

    greed = df[
        df["market_sentiment"]=="Greed"
    ]["closedpnl"]

    t,p = stats.ttest_ind(
        fear,
        greed,
        equal_var=False,
        nan_policy="omit"
    )

    return t,p


def generate_report(df):

    stats_dict = descriptive_statistics(df)

    summary = sentiment_statistics(df)

    t,p = t_test(df)

    fear = df[
        df["market_sentiment"]=="Fear"
    ]["closedpnl"]

    greed = df[
        df["market_sentiment"]=="Greed"
    ]["closedpnl"]

    ci_fear = confidence_interval(fear)

    ci_greed = confidence_interval(greed)

    effect = cohen_d(fear, greed)

    lines = []

    lines.append("# Statistical Report\n")

    lines.append("## Overall Statistics\n")

    for k,v in stats_dict.items():

        lines.append(f"- {k}: {v:.4f}" if isinstance(v,float)
                     else f"- {k}: {v}")

    lines.append("\n")

    lines.append("## By Sentiment\n")

    lines.append(summary.to_string())

    lines.append("\n")

    lines.append("## Welch T-Test\n")

    lines.append(f"T Statistic : {t:.4f}")

    lines.append(f"P Value : {p:.6f}")

    lines.append("\n")

    lines.append("## 95% Confidence Interval")

    lines.append(f"Fear : {ci_fear}")

    lines.append(f"Greed : {ci_greed}")

    lines.append("\n")

    lines.append("## Effect Size")

    lines.append(f"Cohen d : {effect:.4f}")

    with open(
        REPORT_DIR/"statistical_report.md",
        "w"
    ) as f:

        f.write("\n".join(lines))

    print("\nStatistical Report Generated")