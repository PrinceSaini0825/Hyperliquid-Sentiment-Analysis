import pandas as pd
import numpy as np

from src.strategy import apply_strategy
from src.risk_metrics import calculate_metrics



def optimize_strategy(
        merged,
        model,
        X_test
):


    print(
        "\n================================"
    )

    print(
        "Stage 12 Strategy Optimization"
    )

    print(
        "================================"
    )


    #
    # Generate model probabilities
    #

    probabilities = (
        model.predict_proba(
            X_test
        )[:,1]
    )


    #
    # Align test data
    #

    test = merged.iloc[
        -len(probabilities):
    ].copy()



    thresholds = np.arange(
        0.40,
        0.95,
        0.05
    )


    results = []



    for threshold in thresholds:


        temp = test.copy()



        #
        # Generate signals
        #

        temp["probability"] = probabilities


        temp["signal"] = np.where(
            temp["probability"] >= threshold,
            1,
            0
        )



        #
        # Strategy return
        #

        temp["strategy_return"] = (

            temp["signal"]
            *
            temp["closedpnl"]

        ) / (
            temp["sizeusd"] * 0.02
        )



        #
        # Clean values
        #

        temp["strategy_return"] = (
            temp["strategy_return"]
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .fillna(0)
            .clip(
                -0.2,
                0.2
            )
        )



        try:


            metrics = calculate_metrics(
                temp["strategy_return"]
            )


            results.append(

                {
                    "Total Return":
                    metrics["Total Return"],


                    "Win Rate":
                    metrics["Win Rate"],


                    "Sharpe Ratio":
                    metrics["Sharpe Ratio"],


                    "Maximum Drawdown":
                    metrics["Maximum Drawdown"],


                    "Threshold":
                    round(
                        threshold,
                        2
                    )

                }

            )


        except Exception as e:


            print(
                "Threshold failed:",
                threshold,
                e
            )



    results = pd.DataFrame(
        results
    )


    print(
        "\nOptimization Results"
    )

    print(
        "--------------------"
    )


    if results.empty:


        print(
            "No valid strategies generated"
        )

        return None



    results = results.sort_values(
        by="Sharpe Ratio",
        ascending=False
    )


    print(
        results
    )



    best = results.iloc[0]


    print(
        "\nBest Strategy"
    )

    print(
        "----------------"
    )


    print(
        best
    )



    results.to_csv(
        "results/strategy_optimization.csv",
        index=False
    )


    print(
        "✓ Stage 12 Complete"
    )


    return best