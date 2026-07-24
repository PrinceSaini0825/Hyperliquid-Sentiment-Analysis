import pandas as pd
import numpy as np

from src.strategy import apply_strategy
from src.risk_metrics import calculate_metrics
from src.performance_report import save_report



def run_backtest(
        merged,
        model,
        X_test,
        y_test
):


    print(
        "\n================================"
    )

    print(
        "Stage 11 Backtesting"
    )

    print(
        "================================"
    )


    #
    # Model probability prediction
    #

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    #
    # Align test data
    #

    test = merged.iloc[
        -len(probabilities):
    ].copy()



    #
    # Apply trading strategy
    #

    test = apply_strategy(
        test,
        probabilities
    )



    #
    # Position sizing
    #
    # Actual traded capital
    #

    test["position_size"] = (
        test["sizeusd"]
        .abs()
    )



    #
    # Net PnL after fees
    #

    test["net_pnl"] = (

        test["signal"]

        *

        (
            test["closedpnl"]
            -
            test["fee"]
        )

    )



    #
    # Strategy Return
    #

    test["strategy_return"] = (

        test["net_pnl"]

        /

        test["position_size"]

    )



    #
    # Clean numerical issues
    #

    test["strategy_return"] = (

        test["strategy_return"]

        .replace(
            [np.inf, -np.inf],
            np.nan
        )

        .fillna(0)

        .clip(
            -0.2,
            0.2
        )

    )



    #
    # Equity Curve
    #

    initial_capital = 10000


    test["trade_pnl"] = (

    test["strategy_return"]

    *
    
    initial_capital

)


    test["equity_curve"] = (

    initial_capital

    +

    test["trade_pnl"].cumsum()

)



    #
    # Print statistics
    #

    print(
        "\nReturn Statistics"
    )

    print(
        "----------------"
    )


    print(
        "Max:",
        test["strategy_return"].max()
    )


    print(
        "Min:",
        test["strategy_return"].min()
    )


    print(
        "Mean:",
        test["strategy_return"].mean()
    )


    print(
        "Final Equity:",
        test["equity_curve"].iloc[-1]
    )



    #
    # Risk Metrics
    #

    metrics = calculate_metrics(
        test["strategy_return"]
    )



    #
    # Save trading signals
    #

    test.to_csv(
        "results/trading_signals.csv",
        index=False
    )



    #
    # Save equity curve
    #

    test[
        [
            "timestamp",
            "coin",
            "signal",
            "strategy_return",
            "equity_curve"
        ]

    ].to_csv(
        "results/equity_curve.csv",
        index=False
    )



    #
    # Save performance report
    #

    save_report(
        metrics
    )



    print(
        "✓ Backtest Report Saved"
    )


    print(
        metrics
    )


    print(
        "✓ Stage 11 Complete"
    )


    return test