import numpy as np
import pandas as pd



def simulate_portfolio(
        returns,
        initial_capital=10000
):


    equity = [
        initial_capital
    ]


    for r in returns:


        new_value = (
            equity[-1]
            *
            (1+r)
        )

        equity.append(
            new_value
        )


    equity = np.array(
        equity[1:]
    )


    return equity