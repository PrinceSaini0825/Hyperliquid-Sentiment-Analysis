"""
Hyperliquid Sentiment Analysis AI Dashboard

Stage 16.5:
- Chatbot Integration
- Analytics Dashboard
- AI Trading Insights
- UI Polish
- Demo Preparation
"""


import streamlit as st
import pandas as pd
import plotly.express as px


from src.chatbot.agent import (
    ask_llm,
    latest_signals
)


from src.chatbot.tools import (
    load_project_data,
    sentiment_distribution
)



# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(

    page_title="Hyperliquid Sentiment AI",

    page_icon="🚀",

    layout="wide"

)



# -------------------------------------------------
# Custom Styling
# -------------------------------------------------

st.markdown(
"""
<style>


.main-title {

    font-size:42px;

    font-weight:700;

}


.subtitle {

    font-size:18px;

    color:#6b7280;

}


div[data-testid="metric-container"] {

    background-color:#111827;

    padding:15px;

    border-radius:15px;

}


</style>

""",
unsafe_allow_html=True
)



# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title(
    "🚀 Hyperliquid AI"
)


st.sidebar.markdown(
"""
### Project

**Hyperliquid Sentiment Analysis**

AI platform for:

✅ Market Sentiment  
✅ Trader Analytics  
✅ ML Signals  
✅ Risk Analysis  
✅ AI Research Assistant


---

"""
)



st.sidebar.divider()



page = st.sidebar.selectbox(

    "Navigation",

    [

        "🏠 Overview",

        "🧠 Sentiment Analysis",

        "💹 Trader Performance",

        "🤖 Trading Signals",

        "💬 AI Chatbot"

    ]

)



# =================================================
# OVERVIEW
# =================================================


if page == "🏠 Overview":


    st.markdown(
        "<div class='main-title'>🚀 Hyperliquid Sentiment Intelligence Platform</div>",
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class='subtitle'>
        AI-powered crypto market analysis using
        Fear & Greed sentiment, trader behaviour,
        machine learning signals and LLM reasoning.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.divider()



    historical, fear = load_project_data()



    total_trades = len(
        historical
    )


    total_pnl = round(

        historical["closedpnl"]
        .sum(),

        2

    )


    avg_pnl = round(

        historical["closedpnl"]
        .mean(),

        2

    )


    sentiment = (

        fear["classification"]
        .value_counts()

    )


    dominant = sentiment.idxmax()



    col1,col2,col3,col4 = st.columns(4)



    with col1:

        st.metric(
            "📊 Total Trades",
            f"{total_trades:,}"
        )


    with col2:

        st.metric(
            "💰 Total PnL",
            f"${total_pnl}"
        )


    with col3:

        st.metric(
            "📈 Average PnL",
            f"${avg_pnl}"
        )


    with col4:

        st.metric(
            "🧠 Sentiment",
            dominant
        )



    st.divider()



    col1,col2 = st.columns(2)



    with col1:

        st.subheader(
            "Fear & Greed Distribution"
        )


        sentiment_df = (

            sentiment
            .reset_index()

        )


        sentiment_df.columns=[

            "Sentiment",

            "Count"

        ]


        fig = px.pie(

            sentiment_df,

            names="Sentiment",

            values="Count",

            hole=0.45

        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )



    with col2:

        st.subheader(
            "Trader PnL Distribution"
        )


        fig2 = px.histogram(

            historical,

            x="closedpnl",

            nbins=50

        )


        st.plotly_chart(

            fig2,

            use_container_width=True

        )





# =================================================
# SENTIMENT
# =================================================


elif page == "🧠 Sentiment Analysis":


    st.title(
        "🧠 Market Sentiment Analysis"
    )


    data = sentiment_distribution()



    df = pd.DataFrame(

        list(data.items()),

        columns=[

            "Sentiment",

            "Count"

        ]

    )



    fig = px.bar(

        df,

        x="Sentiment",

        y="Count",

        text="Count"

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )


    st.json(data)





# =================================================
# PERFORMANCE
# =================================================


elif page == "💹 Trader Performance":


    st.title(
        "💹 Trader Performance Analytics"
    )


    historical,_ = load_project_data()



    wins = int(

        (

            historical["closedpnl"] > 0

        ).sum()

    )


    losses = int(

        (

            historical["closedpnl"] < 0

        ).sum()

    )


    win_rate = round(

        wins / len(historical) * 100,

        2

    )



    c1,c2,c3 = st.columns(3)



    c1.metric(

        "Winning Trades",

        wins

    )


    c2.metric(

        "Losing Trades",

        losses

    )


    c3.metric(

        "Win Rate",

        f"{win_rate}%"

    )



    st.divider()



    coins = (

        historical["coin"]

        .value_counts()

        .head(10)

        .reset_index()

    )


    coins.columns=[

        "Coin",

        "Trades"

    ]



    fig = px.bar(

        coins,

        x="Coin",

        y="Trades"

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )





# =================================================
# SIGNALS
# =================================================


elif page == "🤖 Trading Signals":


    st.title(
        "🤖 AI Trading Signals"
    )


    signals = latest_signals()



    if isinstance(signals,dict):


        c1,c2,c3 = st.columns(3)


        c1.metric(

            "BUY",

            signals["BUY Signals"]

        )


        c2.metric(

            "SELL",

            signals["SELL Signals"]

        )


        c3.metric(

            "HOLD",

            signals["HOLD Signals"]

        )


        st.divider()


        st.json(signals)



    else:

        st.warning(signals)





# =================================================
# CHATBOT
# =================================================


elif page == "💬 AI Chatbot":


    st.title(
        "💬 Hyperliquid AI Analyst"
    )


    st.caption(

        "Ask questions about sentiment, traders, PnL and signals"

    )



    if "messages" not in st.session_state:

        st.session_state.messages=[]



    for message in st.session_state.messages:


        with st.chat_message(

            message["role"]

        ):

            st.markdown(

                message["content"]

            )



    prompt = st.chat_input(

        "Ask Hyperliquid AI..."

    )



    if prompt:


        st.session_state.messages.append(

            {

            "role":"user",

            "content":prompt

            }

        )



        with st.chat_message("user"):

            st.markdown(prompt)



        with st.chat_message("assistant"):


            with st.spinner(
                "Analyzing market data..."
            ):


                response = ask_llm(prompt)



            st.markdown(response)



        st.session_state.messages.append(

            {

            "role":"assistant",

            "content":response

            }

        )



    if st.sidebar.button(

        "🗑 Clear Chat"

    ):


        st.session_state.messages=[]

        st.rerun()



# -------------------------------------------------
# Footer
# -------------------------------------------------

st.divider()


st.caption(

"""
Hyperliquid Sentiment Analysis AI |
Machine Learning + LLM Analytics Project |
Built with Python, Streamlit & AI Models
"""

)