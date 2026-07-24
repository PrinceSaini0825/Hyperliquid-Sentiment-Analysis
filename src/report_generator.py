"""
Stage 15
Professional Final Report Generator

Generates a Markdown report summarizing the complete ML pipeline.
"""

from pathlib import Path
from datetime import datetime
import pandas as pd


def _metric(value, digits=4):
    try:
        return round(float(value), digits)
    except Exception:
        return value


def generate_final_report(
    stage9_results,
    evaluation_metrics,
    backtest_results,
    strategy_results,
    risk_results,
):
    print("\n================================")
    print("Generating Final Report")
    print("================================")

    output_dir = Path("results/final_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "Final_Report.md"

    lines = []

    lines.append("# Hyperliquid Sentiment Analysis\n")
    lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    lines.append("---\n")

    lines.append("## Pipeline\n")

    stages = [
        "Stage 1  Data Loading",
        "Stage 2  Data Cleaning",
        "Stage 3  Data Merge",
        "Stage 4  EDA",
        "Stage 5  Feature Engineering",
        "Stage 6  ML Dataset",
        "Stage 7  Baseline Models",
        "Stage 8  Training",
        "Stage 9  Model Optimization",
        "Stage 10 Trader Intelligence",
        "Stage 11 Backtesting",
        "Stage 12 Strategy Optimization",
        "Stage 13 Risk Engine",
        "Stage 14 Portfolio Dashboard",
        "Stage 15 Live Trading Signals",
    ]

    for s in stages:
        lines.append(f"- {s}")

    lines.append("\n---\n")

    lines.append("## Model Comparison\n")

    if isinstance(stage9_results, pd.DataFrame):
        lines.append(stage9_results.to_markdown(index=False))
    else:
        lines.append(str(stage9_results))

    lines.append("\n---\n")

    lines.append("## Evaluation Metrics\n")

    if isinstance(evaluation_metrics, dict):
        for k, v in evaluation_metrics.items():
            lines.append(f"- **{k}** : {_metric(v)}")
    else:
        lines.append(str(evaluation_metrics))

    lines.append("\n---\n")

    lines.append("## Best Strategy\n")

    if isinstance(strategy_results, pd.Series):
        for k, v in strategy_results.items():
            lines.append(f"- **{k}** : {_metric(v)}")
    elif isinstance(strategy_results, dict):
        for k, v in strategy_results.items():
            lines.append(f"- **{k}** : {_metric(v)}")
    else:
        lines.append(str(strategy_results))

    lines.append("\n---\n")

    lines.append("## Backtesting\n")

    if isinstance(backtest_results, dict):
        for k, v in backtest_results.items():
            lines.append(f"- **{k}** : {_metric(v)}")
    elif isinstance(backtest_results, pd.DataFrame):
        lines.append(backtest_results.to_markdown(index=False))
    else:
        lines.append(str(backtest_results))

    lines.append("\n---\n")

    lines.append("## Risk Engine\n")

    if isinstance(risk_results, pd.DataFrame):
        lines.append(risk_results.to_markdown(index=False))
    elif isinstance(risk_results, dict):
        for k, v in risk_results.items():
            lines.append(f"- **{k}** : {_metric(v)}")
    else:
        lines.append(str(risk_results))

    lines.append("\n---\n")

    lines.append("## Files Generated\n")

    folders = [
        "results/charts",
        "results/models",
        "results/backtest",
        "results/portfolio",
        "results/signals",
    ]

    for folder in folders:
        p = Path(folder)
        if p.exists():
            files = sorted(p.glob("*"))
            if files:
                lines.append(f"\n### {folder}")
                for f in files:
                    lines.append(f"- {f.name}")

    lines.append("\n---\n")

    lines.append("## Conclusion\n")

    lines.append(
        "The pipeline successfully completed data preprocessing, "
        "feature engineering, model optimization, explainability, "
        "strategy optimization, backtesting, risk analysis, "
        "portfolio analytics and live trading signal generation."
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("✓ Final Report Generated")
    print(report_path)

    return report_path