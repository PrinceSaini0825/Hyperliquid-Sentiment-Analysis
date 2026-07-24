"""
Stage 14 (Part 2A)
Portfolio Visualization

Generates:
1. Equity Curve
2. Drawdown Curve
3. Portfolio CSV Export
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("results/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_portfolio_dashboard(backtest):

    print("\n================================")
    print("Stage 14 Portfolio Visualization")
    print("================================")

    df = backtest.copy()

    # --------------------------------------------------
    # Strategy Return
    # --------------------------------------------------

    if "strategy_return" not in df.columns:
        raise ValueError(
            "strategy_return column not found."
        )

    df["strategy_return"] = (
        pd.to_numeric(
            df["strategy_return"],
            errors="coerce"
        )
        .fillna(0)
        .clip(-0.20, 0.20)
    )

    # --------------------------------------------------
    # Equity Curve
    # --------------------------------------------------

    df["equity"] = (
        10000
        * (1 + df["strategy_return"]).cumprod()
    )

    # --------------------------------------------------
    # Drawdown
    # --------------------------------------------------

    rolling_max = (
        df["equity"]
        .cummax()
    )

    df["drawdown"] = (
        df["equity"] - rolling_max
    ) / rolling_max

    # --------------------------------------------------
    # Export CSV
    # --------------------------------------------------

    export_cols = [
        c for c in [
            "timestamp",
            "market_sentiment",
            "signal",
            "strategy_return",
            "equity",
            "drawdown"
        ]
        if c in df.columns
    ]

    df[export_cols].to_csv(
        OUTPUT_DIR / "portfolio_history.csv",
        index=False
    )

    # --------------------------------------------------
    # Equity Curve
    # --------------------------------------------------

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["equity"],
        linewidth=2
    )

    plt.title(
        "Portfolio Equity Curve",
        fontsize=15,
        weight="bold"
    )

    plt.xlabel("Trade")

    plt.ylabel("Portfolio Value ($)")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "equity_curve.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # Drawdown Curve
    # --------------------------------------------------

    plt.figure(figsize=(14, 5))

    plt.fill_between(
        np.arange(len(df)),
        df["drawdown"],
        0,
        alpha=0.35
    )

    plt.title(
        "Portfolio Drawdown",
        fontsize=15,
        weight="bold"
    )

    plt.xlabel("Trade")

    plt.ylabel("Drawdown")

    plt.grid(alpha=0.30)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "drawdown_curve.png",
        dpi=300
    )

    plt.close()

    print("✓ Equity Curve Saved")
    print("✓ Drawdown Curve Saved")
    print("✓ Portfolio History CSV Saved")

    return df
"""
Part 2B
Rolling Sharpe + Rolling Volatility
"""

import os

import numpy as np
import matplotlib.pyplot as plt


def plot_rolling_metrics(
    portfolio,
    window=30
):
    """
    Rolling Sharpe Ratio and Rolling Volatility.
    """

    os.makedirs(
        "results/portfolio",
        exist_ok=True
    )

    if "strategy_return" not in portfolio.columns:
        print("strategy_return column not found.")
        return

    returns = (
        portfolio["strategy_return"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    rolling_mean = (
        returns
        .rolling(window)
        .mean()
    )

    rolling_std = (
        returns
        .rolling(window)
        .std()
    )

    rolling_sharpe = (
        rolling_mean /
        (rolling_std + 1e-9)
    ) * np.sqrt(window)

    rolling_volatility = (
        rolling_std
        * np.sqrt(window)
    )

    # -------------------------
    # Rolling Sharpe
    # -------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        rolling_sharpe,
        linewidth=2
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.title(
        f"Rolling Sharpe Ratio ({window})"
    )

    plt.xlabel("Trades")
    plt.ylabel("Sharpe")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/portfolio/rolling_sharpe.png",
        dpi=300
    )

    plt.close()

    # -------------------------
    # Rolling Volatility
    # -------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        rolling_volatility,
        linewidth=2
    )

    plt.title(
        f"Rolling Volatility ({window})"
    )

    plt.xlabel("Trades")
    plt.ylabel("Volatility")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/portfolio/rolling_volatility.png",
        dpi=300
    )

    plt.close()

    print("✓ Rolling Metrics Saved")
    """
Part 2C
Return Distribution + Monthly Heatmap
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_return_distribution(portfolio):
    """
    Plot histogram of strategy returns.
    """

    os.makedirs(
        "results/portfolio",
        exist_ok=True
    )

    if "strategy_return" not in portfolio.columns:
        print("strategy_return column not found.")
        return

    returns = (
        portfolio["strategy_return"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    plt.figure(figsize=(10, 5))

    plt.hist(
        returns,
        bins=50
    )

    plt.title("Strategy Return Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/portfolio/return_distribution.png",
        dpi=300
    )

    plt.close()

    print("✓ Return Distribution Saved")


def create_monthly_return_table(portfolio):
    """
    Compute monthly compounded returns and
    save as CSV.
    """

    os.makedirs(
        "results/portfolio",
        exist_ok=True
    )

    required = {
        "timestamp",
        "strategy_return"
    }

    if not required.issubset(portfolio.columns):
        print("timestamp or strategy_return missing.")
        return None

    df = portfolio.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    df["Year"] = df["timestamp"].dt.year
    df["Month"] = df["timestamp"].dt.month_name()

    monthly = (
        df.groupby(
            ["Year", "Month"]
        )["strategy_return"]
        .apply(
            lambda x: (1 + x).prod() - 1
        )
        .reset_index()
    )

    month_order = [
        "January", "February", "March",
        "April", "May", "June",
        "July", "August", "September",
        "October", "November", "December"
    ]

    monthly["Month"] = pd.Categorical(
        monthly["Month"],
        categories=month_order,
        ordered=True
    )

    pivot = monthly.pivot(
        index="Year",
        columns="Month",
        values="strategy_return"
    )

    pivot.to_csv(
        "results/portfolio/monthly_returns.csv"
    )

    print("✓ Monthly Returns CSV Saved")

    return pivot


def plot_monthly_heatmap(portfolio):
    """
    Create a monthly returns heatmap using matplotlib.
    """

    pivot = create_monthly_return_table(
        portfolio
    )

    if pivot is None or pivot.empty:
        print("No monthly return data available.")
        return

    data = (
        pivot.fillna(0)
        .to_numpy()
    )

    plt.figure(figsize=(14, 5))

    image = plt.imshow(
        data,
        aspect="auto"
    )

    plt.colorbar(
        image,
        label="Monthly Return"
    )

    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(pivot.index)),
        pivot.index
    )

    plt.title("Monthly Return Heatmap")

    plt.tight_layout()

    plt.savefig(
        "results/portfolio/monthly_heatmap.png",
        dpi=300
    )

    plt.close()

    print("✓ Monthly Heatmap Saved")
    """
=========================================================
Part 2D
Master Visualization Runner
=========================================================
"""

import os


def generate_portfolio_visualizations(portfolio):

    print("\n================================")
    print("Generating Portfolio Dashboard")
    print("================================")

    os.makedirs(
        "results/portfolio",
        exist_ok=True
    )

    # -------------------------
    # Part 2A
    # -------------------------

    try:
        portfolio = generate_portfolio_dashboard(portfolio)
    except Exception as e:
        print(f"Portfolio Dashboard Error : {e}")

    # -------------------------
    # Part 2B
    # -------------------------

    try:
        plot_rolling_metrics(
            portfolio,
            window=30
        )
    except Exception as e:
        print(f"Rolling Metrics Error : {e}")

    # -------------------------
    # Part 2C
    # -------------------------

    try:
        plot_return_distribution(portfolio)
    except Exception as e:
        print(f"Return Distribution Error : {e}")

    try:
        plot_monthly_heatmap(portfolio)
    except Exception as e:
        print(f"Monthly Heatmap Error : {e}")

    print("\n✓ Portfolio Dashboard Generated")
    print("Location : results/portfolio/")

    return portfolio