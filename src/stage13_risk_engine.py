import numpy as np
import pandas as pd

from src.portfolio_simulator import simulate_portfolio
from src.monte_carlo import monte_carlo_simulation



def run_risk_engine(
        backtest_results
):


    print(
        "\n================================"
    )

    print(
        "Stage 13 Risk Engine"
    )

    print(
        "================================"
    )


    returns = (
        backtest_results[
            "strategy_return"
        ]
        .values
    )



    #
    # Equity Curve
    #

    equity = simulate_portfolio(
        returns
    )


    backtest_results[
        "equity"
    ] = equity



    #
    # Drawdown
    #

    peak = np.maximum.accumulate(
        equity
    )


    drawdown = (
        equity - peak
    ) / peak



    max_drawdown = (
        drawdown.min()
    )



    #
    # Volatility
    #

    volatility = (
        np.std(returns)
        *
        np.sqrt(252)
    )



    #
    # Monte Carlo
    #

    mc_report, mc_values = (
        monte_carlo_simulation(
            returns
        )
    )



    risk_report = {


        "Final Equity":
        equity[-1],


        "Maximum Drawdown":
        max_drawdown,


        "Annual Volatility":
        volatility,


        "Monte Carlo Mean":
        mc_report[
            "Mean Return"
        ],


        "Monte Carlo Median":
        mc_report[
            "Median Return"
        ],


        "Profit Probability":
        mc_report[
            "Probability Profit"
        ]

    }



    pd.DataFrame(
        [risk_report]
    ).to_csv(
        "results/risk_report.csv",
        index=False
    )



    print(
        pd.DataFrame(
            [risk_report]
        )
    )


    print(
        "✓ Stage 13 Complete"
    )


    return risk_report