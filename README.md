# 🚀 Hyperliquid Sentiment Analysis
### AI-Powered Cryptocurrency Trading Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green)]()
[![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-brightgreen)]()
[![CatBoost](https://img.shields.io/badge/CatBoost-Enabled-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-blue)]()
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

---

## 📌 Overview

This project develops a complete machine learning pipeline that analyzes cryptocurrency trader behavior under different market sentiment conditions using **Hyperliquid historical trading data** and the **Bitcoin Fear & Greed Index**.

The pipeline automatically performs

- Data ingestion
- Data preprocessing
- Feature engineering
- Statistical analysis
- Machine learning
- Explainability (SHAP)
- Strategy optimization
- Backtesting
- Risk analysis
- Portfolio analytics
- Live trading signal generation
- Professional report generation

---

# 🎯 Project Goals

- Understand how market sentiment affects trader profitability.
- Engineer trader intelligence features.
- Train multiple ML models.
- Optimize prediction thresholds.
- Evaluate trading strategies.
- Generate explainable trading signals.
- Build a production-ready quantitative research pipeline.

---

# 📊 Dataset

### Hyperliquid Historical Trades

Contains over

> **211,224 historical cryptocurrency trades**

Features include

- Coin
- Trade Size
- Execution Price
- Closed PnL
- Fee
- Position Size
- Timestamp
- Trade Direction
- Order Information

---

### Fear & Greed Index

Contains

> **2,644 daily sentiment observations**

Sentiment Classes

- Extreme Fear
- Fear
- Neutral
- Greed
- Extreme Greed

---

# 🏗 Pipeline Architecture

```
Stage 1
Data Loading
        │
        ▼
Stage 2
Preprocessing
        │
        ▼
Stage 3
Data Merge
        │
        ▼
Stage 4
EDA & Visualization
        │
        ▼
Stage 5
Feature Engineering
        │
        ▼
Stage 6
ML Dataset
        │
        ▼
Stage 7
Baseline Models
        │
        ▼
Stage 8
Training
        │
        ▼
Stage 9
Advanced Optimization
        │
        ▼
Stage 10
Trader Intelligence
        │
        ▼
Stage 11
Backtesting
        │
        ▼
Stage 12
Strategy Optimization
        │
        ▼
Stage 13
Risk Engine
        │
        ▼
Stage 14
Portfolio Analytics
        │
        ▼
Stage 15
Live Trading Signal Engine
```

---

# ⚙️ Machine Learning Models

✔ Logistic Regression

✔ Random Forest

✔ XGBoost

✔ LightGBM

✔ CatBoost

---

# 📈 Final Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|-----------|------------|----------|---------|---------|
| XGBoost | **69.64%** | **61.38%** | **77.13%** | **68.36%** | **0.7793** |
| LightGBM | 68.79% | 60.15% | 78.79% | 68.22% | 0.7736 |
| CatBoost | 64.67% | 55.87% | 80.43% | 65.94% | 0.7380 |

---

# 📊 Evaluation

Accuracy

67.72%

Precision

58.80%

Recall

80.49%

F1 Score

67.95%

ROC-AUC

0.7793

---

# 🔥 Feature Engineering

The project generates more than **30 engineered features**, including

- Historical Trader PnL
- Trader Win Rate
- Average Position Size
- Coin Frequency
- Rolling Profit
- Rolling Volume
- Fee Ratio
- Trade Count
- Historical Trade Size
- Price Change
- Market Sentiment Encoding
- Risk Features

---

# 🧠 Explainable AI

The pipeline integrates SHAP explainability.

Top Features

- Fee Ratio
- Rolling PnL
- Trader Trade Count
- Coin
- Start Position
- Execution Price
- Rolling Trade Size
- Historical Average Trade Size

---

# 📉 Strategy Optimization

Optimized Probability Threshold

```
0.60
```

Sharpe Ratio

```
7.916
```

Maximum Drawdown

```
-8.91%
```

---

# 📈 Backtesting

Total Return

```
380.72%
```

Sharpe Ratio

```
4.52
```

Maximum Drawdown

```
-8.48%
```

---

# 📡 Live Trading Signals

Automatically predicts

- BUY
- SELL
- HOLD

Generates

- Confidence Score
- Position Size
- Risk Level
- Trade Recommendation

Example

```
Coin

HYPE

Signal

BUY

Probability

69.14%

Confidence

38.29%
```

---

# 📂 Project Structure

```
Hyperliquid-Sentiment-Analysis

├── data/
├── models/
├── notebooks/
├── results/
│
├── src/
│
│── loader.py
│── preprocess.py
│── merger.py
│── eda.py
│── visualization.py
│── features.py
│── model.py
│── train.py
│── evaluate.py
│── validation.py
│── hypothesis_test.py
│── shap_explain.py
│── stage9_model_optimization.py
│── stage10_features.py
│── stage11_backtesting.py
│── stage12_strategy_optimizer.py
│── stage13_risk_engine.py
│── portfolio_visualization.py
│── stage15_signal_engine.py
│── report_generator.py
│
└── main.py
```

---

# 📦 Outputs

The pipeline automatically generates

- Statistical Reports
- EDA Charts
- Feature Importance
- SHAP Analysis
- ROC Curve
- Confusion Matrix
- Backtesting Report
- Strategy Optimization Report
- Portfolio Dashboard
- Risk Report
- Trading Signals
- Final Markdown Report

---

# 🛠 Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- Matplotlib
- SciPy
- Joblib

---

# 🚀 Future Work

Planned enhancements include:

- Stage 16: MLOps & Deployment (FastAPI, Streamlit, Docker)
- Stage 17: Deep Learning (LSTM, Transformer, TFT)
- Stage 18: Reinforcement Learning Trading Agent
- Stage 19: Cloud Deployment with CI/CD
- Stage 20: Research Paper & Portfolio Website

---

# 👨‍💻 Author

**Prince Saini**

B.Tech Computer Science

AI • Machine Learning • Quantitative Finance • Data Science • Research Engineering

---

## ⭐ If you found this project useful, consider giving it a star!