"""
Hyperliquid Sentiment Analysis Dashboard
Professional Version
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hyperliquid Sentiment Analysis",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>

.metric{
    background:#F5F5F5;
    padding:15px;
    border-radius:10px;
    text-align:center;
}

.metric h2{
    color:#0068C9;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
RESULTS = Path("results")
FIGURES = Path("figures")
MODELS = Path("models")

df = pd.read_csv(RESULTS / "feature_engineered.csv")

model = joblib.load(MODELS / "best_model.pkl")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    [
        "Overview",
        "Dataset",
        "Visualizations",
        "Model",
        "Prediction"
    ]
)

# ======================================================
# OVERVIEW
# ======================================================

if page == "Overview":

    st.title("Hyperliquid Sentiment Analysis Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Trades", f"{len(df):,}")

    c2.metric("Coins", df["coin"].nunique())

    c3.metric(
        "Average PnL",
        f"${df['closedpnl'].mean():.2f}"
    )

    c4.metric(
        "Profit %",
        f"{100*df['profit_flag'].mean():.2f}%"
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

# ======================================================
# DATASET
# ======================================================

elif page == "Dataset":

    st.title("Dataset Explorer")

    coin = st.selectbox(
        "Coin",
        sorted(df["coin"].unique())
    )

    filtered = df[df["coin"] == coin]

    st.write(filtered)

# ======================================================
# VISUALIZATIONS
# ======================================================

elif page == "Visualizations":

    st.title("Charts")

    charts = sorted(FIGURES.glob("*.png"))

    for chart in charts:

        st.image(
            chart,
            caption=chart.stem,
            use_container_width=True
        )

# ======================================================
# MODEL
# ======================================================

elif page == "Model":

    st.title("Machine Learning Model")

    metrics = pd.read_csv(
        RESULTS / "metrics.csv"
    )

    st.dataframe(
        metrics,
        use_container_width=True
    )

    fi = FIGURES / "feature_importance.png"

    if fi.exists():
        st.image(fi, use_container_width=True)

# ======================================================
# PREDICTION
# ======================================================

else:

    st.title("Live Prediction")

    st.info(
        "This page can be connected to the trained Random Forest model "
        "for real-time predictions."
    )

    st.success(
        "Model Loaded Successfully!"
    )

    st.write(model)