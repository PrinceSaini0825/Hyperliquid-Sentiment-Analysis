"""
Statistical Hypothesis Testing Module

Tests:
- Does market sentiment influence profitability?
- Sentiment group comparison
- Correlation analysis
- Confidence intervals
"""

from pathlib import Path

import pandas as pd
import numpy as np

from scipy.stats import (
    kruskal,
    f_oneway,
    pearsonr,
    spearmanr
)


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)



def confidence_interval(data, confidence=0.95):

    data = np.array(data)

    mean = np.mean(data)

    std_error = (
        np.std(data, ddof=1)
        /
        np.sqrt(len(data))
    )

    margin = 1.96 * std_error

    return (
        mean - margin,
        mean + margin
    )



def run_hypothesis_tests(df):

    print("\nRunning Statistical Hypothesis Tests...")

    results = {}


    # =====================================================
    # Profitability by Sentiment
    # =====================================================

    sentiment_profit = (
        df.groupby("market_sentiment")
        ["profit_flag"]
        .mean()
        .sort_values(
            ascending=False
        )
    )


    sentiment_profit.to_csv(
        RESULTS_DIR /
        "sentiment_profitability.csv"
    )


    print("\nProfit Probability by Sentiment")
    print(sentiment_profit)



    # =====================================================
    # Kruskal Wallis
    # =====================================================

    groups = []

    for _, group in df.groupby(
        "market_sentiment"
    ):

        if len(group) > 10:
            groups.append(
                group["profit_flag"]
            )


    if len(groups) > 1:

        stat, p = kruskal(
            *groups
        )

        results["kruskal_statistic"] = stat
        results["kruskal_p_value"] = p


        print("\nKruskal-Wallis Test")

        print(
            f"Statistic : {stat:.4f}"
        )

        print(
            f"P-value   : {p:.6f}"
        )



    # =====================================================
    # ANOVA
    # =====================================================

    if len(groups) > 1:

        stat, p = f_oneway(
            *groups
        )

        results["anova_statistic"] = stat
        results["anova_p_value"] = p


        print("\nANOVA Test")

        print(
            f"Statistic : {stat:.4f}"
        )

        print(
            f"P-value   : {p:.6f}"
        )



    # =====================================================
    # Correlation Analysis
    # =====================================================

    if "fear_greed_value" in df.columns:

        corr_df = df[
            [
                "fear_greed_value",
                "profit_flag"
            ]
        ].dropna()


        print("\nCorrelation Analysis")

        print(
            f"Samples used : {len(corr_df)}"
        )


        if (
            len(corr_df) > 2
            and
            corr_df["fear_greed_value"].nunique() > 1
        ):


            pearson = pearsonr(
                corr_df["fear_greed_value"],
                corr_df["profit_flag"]
            )


            spearman = spearmanr(
                corr_df["fear_greed_value"],
                corr_df["profit_flag"]
            )


            results["pearson_correlation"] = pearson.statistic
            results["pearson_p_value"] = pearson.pvalue


            results["spearman_correlation"] = spearman.statistic
            results["spearman_p_value"] = spearman.pvalue



            print(
                f"Pearson r : {pearson.statistic:.4f}"
            )

            print(
                f"Pearson p : {pearson.pvalue:.6f}"
            )


            print(
                f"Spearman r: {spearman.statistic:.4f}"
            )

            print(
                f"Spearman p: {spearman.pvalue:.6f}"
            )


        else:

            print(
                "Not enough variation for correlation"
            )



    # =====================================================
    # Confidence Intervals
    # =====================================================


    ci_results = []


    for sentiment, group in df.groupby(
        "market_sentiment"
    ):

        low, high = confidence_interval(
            group["profit_flag"]
        )


        ci_results.append({

            "sentiment": sentiment,

            "profit_probability":
                group["profit_flag"].mean(),

            "CI_lower":
                low,

            "CI_upper":
                high

        })



    pd.DataFrame(
        ci_results
    ).to_csv(
        RESULTS_DIR /
        "sentiment_confidence_intervals.csv",
        index=False
    )



    pd.DataFrame(
        [results]
    ).to_csv(
        RESULTS_DIR /
        "hypothesis_test_results.csv",
        index=False
    )


    print(
        "✓ Hypothesis Testing Complete"
    )


    return results