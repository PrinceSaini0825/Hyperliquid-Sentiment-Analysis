import pandas as pd


def generate_signals(
        probabilities,
        threshold=0.56
):

    signals = []

    for prob in probabilities:

        if prob >= threshold:
            signals.append(1)

        else:
            signals.append(0)


    return signals



def apply_strategy(
        df,
        probabilities,
        threshold=0.56
):

    result = df.copy()


    result["prediction_probability"] = probabilities


    result["signal"] = generate_signals(
        probabilities,
        threshold
    )


    return result