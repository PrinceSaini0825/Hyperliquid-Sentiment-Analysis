import numpy as np



def monte_carlo_simulation(
        returns,
        simulations=10000
):


    results = []


    returns = np.array(
        returns
    )


    for _ in range(simulations):


        sampled = np.random.choice(
            returns,
            size=len(returns),
            replace=True
        )


        cumulative = np.prod(
            1 + sampled
        ) - 1


        results.append(
            cumulative
        )


    results = np.array(
        results
    )


    report = {

        "Simulations":
        simulations,


        "Mean Return":
        np.mean(results),


        "Median Return":
        np.median(results),


        "5% Worst Case":
        np.percentile(
            results,
            5
        ),


        "95% Best Case":
        np.percentile(
            results,
            95
        ),


        "Probability Profit":
        np.mean(results > 0)

    }


    return report, results