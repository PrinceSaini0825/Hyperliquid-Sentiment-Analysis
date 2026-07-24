"""
Stage 14 - Portfolio Analytics

portfolio_metrics.py

Calculates portfolio performance metrics from strategy returns.
"""

import numpy as np
import pandas as pd


# ==========================================================
# Equity Curve
# ==========================================================

def calculate_equity_curve(returns, initial_capital=10000):

    returns = pd.Series(returns).fillna(0)

    equity = initial_capital * (1 + returns).cumprod()

    return equity


# ==========================================================
# Drawdown
# ==========================================================

def calculate_drawdown(equity_curve):

    running_max = equity_curve.cummax()

    drawdown = (
        equity_curve - running_max
    ) / running_max

    return drawdown


# ==========================================================
# CAGR
# ==========================================================

def calculate_cagr(equity_curve, periods_per_year=252):

    n = len(equity_curve)

    if n <= 1:
        return 0.0

    years = n / periods_per_year

    if years <= 0:
        return 0.0

    start = float(equity_curve.iloc[0])
    end = float(equity_curve.iloc[-1])

    if start <= 0 or end <= 0:
        return 0.0

    return (end / start) ** (1 / years) - 1


# ==========================================================
# Annual Return
# ==========================================================

def calculate_annual_return(returns, periods_per_year=252):

    returns = pd.Series(returns).fillna(0)

    return returns.mean() * periods_per_year


# ==========================================================
# Annual Volatility
# ==========================================================

def calculate_volatility(returns, periods_per_year=252):

    returns = pd.Series(returns).fillna(0)

    return returns.std() * np.sqrt(periods_per_year)


# ==========================================================
# Sharpe Ratio
# ==========================================================

def calculate_sharpe(
        returns,
        risk_free_rate=0.0,
        periods_per_year=252
):

    returns = pd.Series(returns).fillna(0)

    excess = returns - risk_free_rate / periods_per_year

    std = excess.std()

    if std == 0:
        return 0.0

    return (
        excess.mean()
        / std
    ) * np.sqrt(periods_per_year)


# ==========================================================
# Sortino Ratio
# ==========================================================

def calculate_sortino(
        returns,
        risk_free_rate=0.0,
        periods_per_year=252
):

    returns = pd.Series(returns).fillna(0)

    downside = returns[returns < 0]

    if len(downside) == 0:
        return 0.0

    downside_std = downside.std()

    if downside_std == 0:
        return 0.0

    excess = returns.mean() - risk_free_rate / periods_per_year

    return (
        excess
        / downside_std
    ) * np.sqrt(periods_per_year)


# ==========================================================
# Maximum Drawdown
# ==========================================================

def calculate_max_drawdown(drawdown):

    if len(drawdown) == 0:
        return 0.0

    return float(drawdown.min())


# ==========================================================
# Calmar Ratio
# ==========================================================

def calculate_calmar(cagr, max_drawdown):

    if max_drawdown == 0:
        return 0.0

    return cagr / abs(max_drawdown)


# ==========================================================
# Recovery Factor
# ==========================================================

def calculate_recovery_factor(
        total_return,
        max_drawdown
):

    if max_drawdown == 0:
        return 0.0

    return total_return / abs(max_drawdown)


# ==========================================================
# Win Rate
# ==========================================================

def calculate_win_rate(returns):

    returns = pd.Series(returns)

    if len(returns) == 0:
        return 0.0

    return (returns > 0).mean()


# ==========================================================
# Profit Factor
# ==========================================================

def calculate_profit_factor(returns):

    returns = pd.Series(returns)

    gross_profit = returns[returns > 0].sum()

    gross_loss = abs(
        returns[returns < 0].sum()
    )

    if gross_loss == 0:
        return np.inf

    return gross_profit / gross_loss


# ==========================================================
# Expectancy
# ==========================================================

def calculate_expectancy(returns):

    returns = pd.Series(returns)

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = calculate_win_rate(returns)

    loss_rate = 1 - win_rate

    avg_win = wins.mean() if len(wins) else 0

    avg_loss = abs(losses.mean()) if len(losses) else 0

    expectancy = (
        win_rate * avg_win
        -
        loss_rate * avg_loss
    )

    return expectancy


# ==========================================================
# Consecutive Wins / Losses
# ==========================================================

def calculate_streaks(returns):

    wins = returns > 0

    max_wins = 0
    max_losses = 0

    current_wins = 0
    current_losses = 0

    for value in wins:

        if value:

            current_wins += 1
            current_losses = 0

        else:

            current_losses += 1
            current_wins = 0

        max_wins = max(
            max_wins,
            current_wins
        )

        max_losses = max(
            max_losses,
            current_losses
        )

    return max_wins, max_losses


# ==========================================================
# Rolling Sharpe
# ==========================================================

def rolling_sharpe(
        returns,
        window=30
):

    returns = pd.Series(returns).fillna(0)

    return (
        returns.rolling(window)
        .mean()
        /
        returns.rolling(window)
        .std()
    ) * np.sqrt(252)


# ==========================================================
# Rolling Volatility
# ==========================================================

def rolling_volatility(
        returns,
        window=30
):

    returns = pd.Series(returns).fillna(0)

    return (
        returns.rolling(window)
        .std()
    ) * np.sqrt(252)


# ==========================================================
# Complete Portfolio Metrics
# ==========================================================

def calculate_portfolio_metrics(
        strategy_returns,
        initial_capital=10000
):

    strategy_returns = (
        pd.Series(strategy_returns)
        .fillna(0)
    )

    equity = calculate_equity_curve(
        strategy_returns,
        initial_capital
    )

    drawdown = calculate_drawdown(
        equity
    )

    cagr = calculate_cagr(equity)

    annual_return = calculate_annual_return(
        strategy_returns
    )

    volatility = calculate_volatility(
        strategy_returns
    )

    sharpe = calculate_sharpe(
        strategy_returns
    )

    sortino = calculate_sortino(
        strategy_returns
    )

    max_dd = calculate_max_drawdown(
        drawdown
    )

    calmar = calculate_calmar(
        cagr,
        max_dd
    )

    total_return = (
        equity.iloc[-1] / equity.iloc[0]
    ) - 1

    recovery = calculate_recovery_factor(
        total_return,
        max_dd
    )

    profit_factor = calculate_profit_factor(
        strategy_returns
    )

    win_rate = calculate_win_rate(
        strategy_returns
    )

    expectancy = calculate_expectancy(
        strategy_returns
    )

    streak_win, streak_loss = calculate_streaks(
        strategy_returns
    )

    summary = {

        "Initial Capital": initial_capital,

        "Final Equity": equity.iloc[-1],

        "Total Return": total_return,

        "CAGR": cagr,

        "Annual Return": annual_return,

        "Annual Volatility": volatility,

        "Sharpe Ratio": sharpe,

        "Sortino Ratio": sortino,

        "Calmar Ratio": calmar,

        "Maximum Drawdown": max_dd,

        "Recovery Factor": recovery,

        "Profit Factor": profit_factor,

        "Win Rate": win_rate,

        "Expectancy": expectancy,

        "Best Trade": strategy_returns.max(),

        "Worst Trade": strategy_returns.min(),

        "Average Trade": strategy_returns.mean(),

        "Average Win":
            strategy_returns[
                strategy_returns > 0
            ].mean(),

        "Average Loss":
            strategy_returns[
                strategy_returns < 0
            ].mean(),

        "Total Trades":
            len(strategy_returns),

        "Consecutive Wins":
            streak_win,

        "Consecutive Losses":
            streak_loss

    }

    return (
        summary,
        equity,
        drawdown,
        rolling_sharpe(strategy_returns),
        rolling_volatility(strategy_returns)
    )