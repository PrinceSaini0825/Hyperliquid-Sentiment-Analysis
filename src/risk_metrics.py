import numpy as np
import pandas as pd



def calculate_metrics(returns):


    returns = (
        returns
        .replace(
            [np.inf,-np.inf],
            np.nan
        )
        .fillna(0)
    )


    #
    # Fixed capital equity simulation
    #

    initial_capital = 10000

    pnl = (
        returns
        *
        initial_capital
    )


    equity = (
        initial_capital
        +
        pnl.cumsum()
    )


    total_return = (

        equity.iloc[-1]
        /
        initial_capital

    ) - 1



    win_rate = (
        returns > 0
    ).mean()



    volatility = (
        returns.std()
    )


    sharpe = (

        returns.mean()
        /
        volatility

        *
        np.sqrt(252)

        if volatility != 0
        else 0

    )



    running_max = (
        equity.cummax()
    )


    drawdown = (

        equity
        /
        running_max

        -
        1

    )


    max_drawdown = (
        drawdown.min()
    )



    return {

        "Total Return": round(
            total_return,
            4
        ),

        "Win Rate": round(
            win_rate,
            4
        ),

        "Sharpe Ratio": round(
            sharpe,
            4
        ),

        "Maximum Drawdown": round(
            max_drawdown,
            4
        )

    }