"""
Stage 15
Live Trading Signal Engine

Generates:
- BUY / HOLD / SELL signals
- Confidence score
- Position sizing
- Trading recommendations
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------
# Output Folder
# -----------------------------------------------------

OUTPUT_DIR = Path("results/signals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------
# Predict Probabilities
# -----------------------------------------------------

def predict_probabilities(model, X):
    """
    Predict probability of profitable trade.

    Returns
    -------
    ndarray
        Probability of class 1
    """

    print("\nPredicting Trade Probabilities...")

    probabilities = model.predict_proba(X)[:, 1]

    print("✓ Probability Prediction Complete")

    return probabilities


# -----------------------------------------------------
# Confidence Score
# -----------------------------------------------------

def calculate_confidence(probabilities):
    """
    Confidence score

    0   = uncertain
    1.0 = extremely confident
    """

    print("\nCalculating Confidence Scores...")

    confidence = np.abs(probabilities - 0.5) * 2

    confidence = np.clip(confidence, 0, 1)

    print("✓ Confidence Scores Generated")

    return confidence


# -----------------------------------------------------
# Confidence Label
# -----------------------------------------------------

def confidence_label(score):
    """
    Convert confidence score into label.
    """

    if score >= 0.90:
        return "Very High"

    elif score >= 0.75:
        return "High"

    elif score >= 0.50:
        return "Medium"

    elif score >= 0.30:
        return "Low"

    return "Very Low"


# -----------------------------------------------------
# Generate Signals
# -----------------------------------------------------

def generate_signals(
    probabilities,
    buy_threshold=0.65,
    sell_threshold=0.40
):
    """
    BUY / HOLD / SELL generation
    """

    print("\nGenerating Trading Signals...")

    signals = []

    for prob in probabilities:

        if prob >= buy_threshold:

            signals.append("BUY")

        elif prob <= sell_threshold:

            signals.append("SELL")

        else:

            signals.append("HOLD")

    print("✓ Signals Generated")

    return np.array(signals)


# -----------------------------------------------------
# Preview Function
# -----------------------------------------------------

def preview_signals(probabilities, confidence, signals, n=10):
    """
    Display first few generated signals.
    """

    print("\nSignal Preview")
    print("-" * 60)

    preview = pd.DataFrame(
        {
            "Probability": np.round(probabilities[:n], 4),
            "Confidence": np.round(confidence[:n], 4),
            "Signal": signals[:n],
            "Confidence Label": [
                confidence_label(x)
                for x in confidence[:n]
            ],
        }
    )

    print(preview)

    return preview
# -----------------------------------------------------
# Apply Risk Filter
# -----------------------------------------------------

def apply_risk_filter(
    signals,
    confidence,
    min_confidence=0.30
):
    """
    Replace low-confidence BUY/SELL signals with HOLD.
    """

    print("\nApplying Risk Filter...")

    filtered = []

    for signal, conf in zip(signals, confidence):

        if conf < min_confidence:
            filtered.append("HOLD")
        else:
            filtered.append(signal)

    filtered = np.array(filtered)

    print("✓ Risk Filter Applied")

    return filtered


# -----------------------------------------------------
# Position Size Recommendation
# -----------------------------------------------------

def recommend_position_size(confidence):
    """
    Recommend capital allocation based on confidence.
    """

    if confidence >= 0.90:
        return "100%"

    elif confidence >= 0.80:
        return "75%"

    elif confidence >= 0.65:
        return "50%"

    elif confidence >= 0.50:
        return "25%"

    return "10%"


# -----------------------------------------------------
# Risk Category
# -----------------------------------------------------

def risk_category(confidence):
    """
    Lower confidence implies higher trading risk.
    """

    if confidence >= 0.90:
        return "Very Low"

    elif confidence >= 0.75:
        return "Low"

    elif confidence >= 0.55:
        return "Medium"

    elif confidence >= 0.35:
        return "High"

    return "Very High"


# -----------------------------------------------------
# Create Signal DataFrame
# -----------------------------------------------------

def create_signal_dataframe(
    merged,
    probabilities,
    confidence,
    signals
):
    """
    Build the master signal DataFrame.
    """

    print("\nCreating Signal DataFrame...")

    n = len(probabilities)

    data = pd.DataFrame({

        "Timestamp":
            merged["timestamp"].tail(n).reset_index(drop=True),

        "Coin":
            merged["coin"].tail(n).reset_index(drop=True),

        "Probability":
            np.round(probabilities, 4),

        "Confidence":
            np.round(confidence, 4),

        "Confidence Label":
            [confidence_label(x) for x in confidence],

        "Signal":
            signals,

        "Position Size":
            [
                recommend_position_size(x)
                for x in confidence
            ],

        "Risk":
            [
                risk_category(x)
                for x in confidence
            ]

    })

    print("✓ Signal DataFrame Created")

    return data


# -----------------------------------------------------
# Save Signals
# -----------------------------------------------------

def save_signal_csv(signal_df):
    """
    Save complete signal history and latest signal.
    """

    signal_df.to_csv(

        OUTPUT_DIR / "signals.csv",

        index=False

    )

    signal_df.tail(1).to_csv(

        OUTPUT_DIR / "latest_signal.csv",

        index=False

    )

    print("✓ Signal CSV Saved")


# -----------------------------------------------------
# Signal Summary
# -----------------------------------------------------

def print_signal_summary(signal_df):
    """
    Print signal counts.
    """

    print("\nSignal Summary")
    print("-" * 50)

    print(signal_df["Signal"].value_counts())

    print("\nAverage Confidence:",
          round(signal_df["Confidence"].mean(), 4))

    print("Highest Confidence:",
          round(signal_df["Confidence"].max(), 4))

    print("Lowest Confidence:",
          round(signal_df["Confidence"].min(), 4))

# -----------------------------------------------------
# Generate Signal Statistics
# -----------------------------------------------------

def generate_statistics(signal_df):
    """
    Generate summary statistics for trading signals.
    """

    print("\nGenerating Signal Statistics...")

    total = len(signal_df)

    buy_count = (signal_df["Signal"] == "BUY").sum()
    sell_count = (signal_df["Signal"] == "SELL").sum()
    hold_count = (signal_df["Signal"] == "HOLD").sum()

    stats = pd.DataFrame({

        "Total Signals": [total],

        "BUY Signals": [buy_count],
        "SELL Signals": [sell_count],
        "HOLD Signals": [hold_count],

        "BUY %": [round(100 * buy_count / total, 2)],
        "SELL %": [round(100 * sell_count / total, 2)],
        "HOLD %": [round(100 * hold_count / total, 2)],

        "Average Confidence":
            [round(signal_df["Confidence"].mean(), 4)],

        "Median Confidence":
            [round(signal_df["Confidence"].median(), 4)],

        "Maximum Confidence":
            [round(signal_df["Confidence"].max(), 4)],

        "Minimum Confidence":
            [round(signal_df["Confidence"].min(), 4)]

    })

    print(stats)

    return stats


# -----------------------------------------------------
# Save Statistics CSV
# -----------------------------------------------------

def save_statistics_csv(stats):

    file_path = OUTPUT_DIR / "signal_statistics.csv"

    stats.to_csv(
        file_path,
        index=False
    )

    print("✓ Signal Statistics Saved")


# -----------------------------------------------------
# Trade Recommendation Generator
# -----------------------------------------------------

def generate_trade_recommendation(signal_df):
    """
    Create a recommendation for the latest signal.
    """

    latest = signal_df.iloc[-1]

    signal = latest["Signal"]
    coin = latest["Coin"]
    confidence = latest["Confidence"]
    risk = latest["Risk"]
    position = latest["Position Size"]
    probability = latest["Probability"]

    if signal == "BUY":

        reason = (
            "Model predicts a high probability of a profitable trade."
        )

    elif signal == "SELL":

        reason = (
            "Model predicts weak trade performance. Avoid long exposure."
        )

    else:

        reason = (
            "Prediction confidence is insufficient for a directional trade."
        )

    recommendation = f"""
==========================================
LATEST TRADING SIGNAL
==========================================

Coin               : {coin}

Signal             : {signal}

Probability        : {probability:.4f}

Confidence         : {confidence:.2%}

Risk               : {risk}

Recommended Size   : {position}

Reason
------
{reason}

Generated Automatically
Stage 15 Trading Signal Engine
"""

    return recommendation


# -----------------------------------------------------
# Save Recommendation
# -----------------------------------------------------

def save_trade_recommendation(signal_df):

    recommendation = generate_trade_recommendation(
        signal_df
    )

    file_path = OUTPUT_DIR / "trade_recommendation.txt"

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(recommendation)

    print("✓ Trade Recommendation Saved")

    print("\n")
    print(recommendation)


# -----------------------------------------------------
# Generate Complete Report
# -----------------------------------------------------

def generate_signal_report(signal_df):
    """
    Save all text/CSV reports.
    """

    stats = generate_statistics(signal_df)

    save_statistics_csv(stats)

    save_trade_recommendation(signal_df)

    return stats

# -----------------------------------------------------
# Visualization Imports
# -----------------------------------------------------

import matplotlib.pyplot as plt


# -----------------------------------------------------
# Probability Curve
# -----------------------------------------------------

def plot_probability_curve(signal_df):
    """
    Plot model probabilities.
    """

    plt.figure(figsize=(14,6))

    plt.plot(
        signal_df["Probability"].values,
        linewidth=1.5
    )

    plt.axhline(
        0.65,
        color="green",
        linestyle="--",
        label="BUY Threshold"
    )

    plt.axhline(
        0.40,
        color="red",
        linestyle="--",
        label="SELL Threshold"
    )

    plt.title("Predicted Trade Probability")

    plt.xlabel("Trade Number")

    plt.ylabel("Probability")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "signal_probability.png",
        dpi=300
    )

    plt.close()

    print("✓ Probability Curve Saved")


# -----------------------------------------------------
# Confidence Distribution
# -----------------------------------------------------

def plot_confidence_distribution(signal_df):
    """
    Histogram of confidence scores.
    """

    plt.figure(figsize=(10,6))

    plt.hist(
        signal_df["Confidence"],
        bins=30
    )

    plt.title("Confidence Distribution")

    plt.xlabel("Confidence")

    plt.ylabel("Count")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confidence_distribution.png",
        dpi=300
    )

    plt.close()

    print("✓ Confidence Distribution Saved")


# -----------------------------------------------------
# Signal Timeline
# -----------------------------------------------------

def plot_signal_timeline(signal_df):
    """
    Timeline of BUY / HOLD / SELL signals.
    """

    mapping = {
        "SELL": -1,
        "HOLD": 0,
        "BUY": 1
    }

    values = signal_df["Signal"].map(mapping)

    plt.figure(figsize=(16,5))

    plt.plot(
        values.values,
        linewidth=1
    )

    plt.yticks(
        [-1,0,1],
        ["SELL","HOLD","BUY"]
    )

    plt.title("Trading Signal Timeline")

    plt.xlabel("Trade Number")

    plt.ylabel("Signal")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "signal_timeline.png",
        dpi=300
    )

    plt.close()

    print("✓ Signal Timeline Saved")


# -----------------------------------------------------
# BUY / SELL / HOLD Bar Chart
# -----------------------------------------------------

def plot_buy_sell_counts(signal_df):
    """
    Bar chart of signal counts.
    """

    counts = signal_df["Signal"].value_counts()

    plt.figure(figsize=(7,5))

    plt.bar(
        counts.index,
        counts.values
    )

    plt.title("Signal Distribution")

    plt.xlabel("Signal")

    plt.ylabel("Count")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "signal_bar_chart.png",
        dpi=300
    )

    plt.close()

    print("✓ Signal Bar Chart Saved")


# -----------------------------------------------------
# Generate All Charts
# -----------------------------------------------------

def generate_signal_plots(signal_df):
    """
    Generate all Stage 15 charts.
    """

    print("\nGenerating Signal Visualizations...")

    plot_probability_curve(signal_df)

    plot_confidence_distribution(signal_df)

    plot_signal_timeline(signal_df)

    plot_buy_sell_counts(signal_df)

    print("✓ All Signal Charts Generated")

# -----------------------------------------------------
# Signal Pie Chart
# -----------------------------------------------------

def plot_signal_pie(signal_df):
    """
    Pie chart showing BUY / HOLD / SELL proportions.
    """

    counts = signal_df["Signal"].value_counts()

    plt.figure(figsize=(7, 7))

    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Signal Distribution")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "signal_pie_chart.png",
        dpi=300
    )

    plt.close()

    print("✓ Signal Pie Chart Saved")


# -----------------------------------------------------
# Export Latest Signal JSON
# -----------------------------------------------------

def export_latest_signal_json(signal_df):
    """
    Export latest signal for dashboards/APIs.
    """

    latest = signal_df.tail(1)

    latest.to_json(
        OUTPUT_DIR / "latest_signal.json",
        orient="records",
        indent=4
    )

    print("✓ Latest Signal JSON Saved")


# -----------------------------------------------------
# Stage 15 Runner
# -----------------------------------------------------

def run_stage15(
    merged,
    model,
    X
):
    """
    Complete Stage 15 pipeline.
    """

    print("\n================================")
    print("Stage 15 Live Trading Signals")
    print("================================")

    # ---------------------------------
    # Prediction
    # ---------------------------------

    probabilities = predict_probabilities(
        model,
        X
    )

    confidence = calculate_confidence(
        probabilities
    )

    signals = generate_signals(
        probabilities
    )

    signals = apply_risk_filter(
        signals,
        confidence
    )

    preview_signals(
        probabilities,
        confidence,
        signals
    )

    # ---------------------------------
    # DataFrame
    # ---------------------------------

    signal_df = create_signal_dataframe(
        merged,
        probabilities,
        confidence,
        signals
    )

    save_signal_csv(
        signal_df
    )

    print_signal_summary(
        signal_df
    )

    # ---------------------------------
    # Reports
    # ---------------------------------

    stats = generate_signal_report(
        signal_df
    )

    # ---------------------------------
    # Charts
    # ---------------------------------

    generate_signal_plots(
        signal_df
    )

    plot_signal_pie(
        signal_df
    )

    export_latest_signal_json(
        signal_df
    )

    print("\n================================")
    print("Stage 15 Complete")
    print("================================")

    print(stats)

    print("\n✓ Signal Dashboard Generated")
    print("Location :", OUTPUT_DIR)

    return signal_df