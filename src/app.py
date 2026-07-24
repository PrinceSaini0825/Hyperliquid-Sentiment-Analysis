import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="Hyperliquid Sentiment Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# Paths
# ---------------------------------------------------

RESULTS = Path("results")
MODELS = Path("models")

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

@st.cache_data
def load_dataframe():
    return pd.read_csv(RESULTS / "feature_engineered.csv")


@st.cache_resource
def load_model():
    return joblib.load(MODELS / "best_model.pkl")


df = load_dataframe()
model = load_model()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.title("Filters")

coin = st.sidebar.multiselect(
    "Coin",
    sorted(df["coin"].dropna().unique()),
    default=sorted(df["coin"].dropna().unique())
)

sentiment = st.sidebar.multiselect(
    "Market Sentiment",
    sorted(df["market_sentiment"].dropna().unique()),
    default=sorted(df["market_sentiment"].dropna().unique())
)

filtered = df[
    df["coin"].isin(coin)
]

filtered = filtered[
    filtered["market_sentiment"].isin(sentiment)
]

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.title("📈 Hyperliquid Sentiment Analysis Dashboard")

st.markdown(
    "Machine Learning analysis of Hyperliquid trader performance under different market sentiments."
)

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Trades",
    f"{len(filtered):,}"
)

c2.metric(
    "Average PnL",
    f"${filtered['closedpnl'].mean():.2f}"
)

c3.metric(
    "Win Rate",
    f"{filtered['profit_flag'].mean()*100:.2f}%"
)

c4.metric(
    "Coins",
    filtered["coin"].nunique()
)

st.divider()

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Charts",
    "Prediction",
    "Dataset"
])

# ===================================================
# TAB 1
# ===================================================

with tab1:

    st.subheader("Market Sentiment Distribution")

    fig = px.pie(
        filtered,
        names="market_sentiment",
        hole=0.45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Profit Distribution")

    fig = px.histogram(
        filtered,
        x="closedpnl",
        nbins=50
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ===================================================
# TAB 2
# ===================================================

with tab2:

    st.subheader("Trade Size")

    fig = px.box(
        filtered,
        x="market_sentiment",
        y="sizeusd",
        color="market_sentiment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Top Coins")

    top = (
        filtered["coin"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top.columns = ["Coin", "Trades"]

    fig = px.bar(
        top,
        x="Coin",
        y="Trades"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ===================================================
# TAB 3
# ===================================================

with tab3:

    st.subheader("Live Prediction")

    st.info(
        "Example prediction interface."
    )

    size = st.number_input(
        "Trade Size (USD)",
        value=1000.0
    )

    fee = st.number_input(
        "Fee",
        value=2.0
    )

    start_position = st.number_input(
        "Start Position",
        value=0.0
    )

    if st.button("Predict"):

        st.success(
            "Prediction module ready.\n\nConnect this to your predict.py for real-time inference."
        )

# ===================================================
# TAB 4
# ===================================================

with tab4:

    st.subheader("Dataset Preview")

    st.dataframe(
        filtered,
        use_container_width=True
    )

    st.download_button(
        "Download CSV",
        filtered.to_csv(index=False),
        "filtered_dataset.csv",
        "text/csv"
    )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Hyperliquid Sentiment Analysis | Machine Learning Dashboard"
)
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