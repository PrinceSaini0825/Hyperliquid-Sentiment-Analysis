"""
Stage 15
Professional Reporting System

Part 2D
Master Report Generator
"""
"""
Stage 15 Reporting Wrapper
"""

"""
Stage 15 Reporting Wrapper
"""

from .report_generator import generate_final_report

__all__ = ["generate_final_report"]


def generate_final_report(
    stage9_results,
    evaluation_metrics,
    backtest_results,
    strategy_results,
    risk_results,
):
    """
    Generates the complete project report.
    """

    print("\n================================")
    print("Stage 15 Professional Reporting")
    print("================================")

    output_dir = Path("results/final_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_summary_report(
        output_dir
    )

    generate_model_report(
        stage9_results,
        evaluation_metrics,
        output_dir
    )

    generate_strategy_report(
        strategy_results,
        output_dir
    )

    generate_risk_report(
        risk_results,
        output_dir
    )

    generate_portfolio_report(
        backtest_results,
        output_dir
    )

    print("✓ Final Report Generated")
    print(f"Location : {output_dir}")

    return output_dir