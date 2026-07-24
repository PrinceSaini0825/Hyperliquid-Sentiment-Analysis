import pandas as pd



def save_report(metrics):


    df = pd.DataFrame(
        [metrics]
    )


    df.to_csv(
        "results/backtest_report.csv",
        index=False
    )


    print(
        "\nBacktest Report Saved"
    )

    print(
        df
    )