from src.loader import load_data
from src.preprocess import preprocess_data
from src.merger import merge_data
from src.eda import run_eda
from src.utils import print_header
from src.statistics import generate_report
from src.visualization import run_visualizations
from src.insights import generate_insights
from src.features import engineer_features
from src.model import prepare_ml_data

from src.train import save_feature_importance
from src.evaluate import evaluate_model
from src.validation import validate_model

from src.experiment import save_experiment
from src.confusion_matrix import save_confusion_matrix
from src.roc_curve import save_roc_curve
from src.shap_explain import generate_shap_analysis
from src.hypothesis_test import run_hypothesis_tests

from src.stage9_model_optimization import train_stage9
from src.stage10_features import add_trader_features
from src.stage11_backtesting import run_backtest
from src.stage12_strategy_optimizer import optimize_strategy
from src.stage13_risk_engine import run_risk_engine
from src.portfolio_visualization import generate_portfolio_visualizations
from src.stage15_signal_engine import run_stage15

from src.report_generator import generate_final_report

import joblib


def main():

    print_header("Hyperliquid Sentiment Analysis")

    # ======================================================
    # Stage 1 : Load Data
    # ======================================================

    historical, fear = load_data()

    # ======================================================
    # Stage 2 : Preprocess
    # ======================================================

    historical, fear = preprocess_data(
        historical,
        fear
    )

    # ======================================================
    # Stage 3 : Merge
    # ======================================================

    merged = merge_data(
        historical,
        fear
    )

    # ======================================================
    # Stage 4 : Analysis
    # ======================================================

    run_eda(merged)
    generate_report(merged)
    run_visualizations(merged)
    generate_insights(merged)

    print("\nMerged Data Types:")
    print(merged.dtypes)

    # ======================================================
    # Stage 5 : Feature Engineering
    # ======================================================

    merged = engineer_features(merged)

    # ======================================================
    # Stage 10 : Trader Intelligence
    # ======================================================

    merged = add_trader_features(merged)

    run_hypothesis_tests(merged)

    # ======================================================
    # Stage 6 : ML Dataset
    # ======================================================

    X_train, X_test, y_train, y_test = prepare_ml_data(
        merged
    )

    print("\nDataset Ready")
    print("Train :", X_train.shape)
    print("Test  :", X_test.shape)

    # ======================================================
    # Stage 9 : Model Optimization
    # ======================================================

    print("\nStage 9/10 Advanced Model Optimization...")

    stage9_results = train_stage9(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nModel Results")
    print(stage9_results)

    best_model_name = stage9_results.iloc[0]["Model"]

    print("\nBest Model:", best_model_name)

    best_model = joblib.load(
        "models/best_model.pkl"
    )

    # ======================================================
    # Evaluation
    # ======================================================

    save_feature_importance(
        best_model,
        X_train
    )

    evaluation_metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    save_confusion_matrix(
        best_model,
        X_test,
        y_test
    )

    save_roc_curve(
        best_model,
        X_test,
        y_test
    )

    generate_shap_analysis(
        best_model,
        X_test
    )

    validate_model(
        best_model,
        X_test,
        y_test
    )

    # ======================================================
    # Stage 11 : Backtesting
    # ======================================================

    backtest_results = run_backtest(
        merged,
        best_model,
        X_test,
        y_test
    )

    # ======================================================
    # Experiment Tracking
    # ======================================================

    selected_metric = stage9_results.iloc[0]

    save_experiment(
        best_model_name,
        selected_metric,
        X_train.shape[1],
        len(X_train),
        len(X_test)
    )

    # ======================================================
    # Stage 12 : Strategy Optimization
    # ======================================================

    strategy_results = optimize_strategy(
        merged,
        best_model,
        X_test
    )

    # ======================================================
    # Stage 13 : Risk Engine
    # ======================================================

    risk_results = run_risk_engine(
        backtest_results
    )

    # ======================================================
    # Stage 14 : Portfolio Visualization
    # ======================================================

    generate_portfolio_visualizations(
        backtest_results
    )

    # ======================================================
    # Stage 15 : Live Signal Engine
    # ======================================================

    signal_df = run_stage15(
        merged,
        best_model,
        X_test
    )

    # ======================================================
    # Final Report
    # ======================================================

    generate_final_report(
        stage9_results=stage9_results,
        evaluation_metrics=evaluation_metrics,
        backtest_results=backtest_results,
        strategy_results=strategy_results,
        risk_results=risk_results,
    )

    print("\n✅ Stage 15 Pipeline Completed Successfully")


if __name__ == "__main__":
    main()