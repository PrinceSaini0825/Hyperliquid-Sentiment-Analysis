import streamlit as st

from src.chatbot.tools import (
    dataset_summary,
)

from src.chatbot.agent import ask

st.set_page_config(
    page_title="Hyperliquid AI Analyst",
    layout="wide"
)

st.title("🤖 Hyperliquid AI Analyst")

question = st.chat_input(
    "Ask about your trading data..."
)

if question:

    context = dataset_summary()

    answer = ask(
        question,
        context
    )

    st.chat_message("assistant").write(answer)