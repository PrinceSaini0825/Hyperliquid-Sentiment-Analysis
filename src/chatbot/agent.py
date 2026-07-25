from langchain_ollama import ChatOllama

from src.chatbot.project_context import PROJECT_CONTEXT
from src.chatbot.tools import (
    dataset_summary,
    pnl_statistics,
    top_coins,
    buy_sell_ratio,
    sentiment_distribution,
    latest_predictions,
    latest_signals,
)

# --------------------------------------------------
# Initialize Local LLM
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0,
)

# --------------------------------------------------
# Main Chat Function
# --------------------------------------------------


def ask_llm(question: str):

    q = question.lower()

    # ==================================================
    # Dataset Summary
    # ==================================================

    if any(word in q for word in ["dataset", "summary", "rows", "columns"]):

        data = dataset_summary()

        prompt = f"""
{PROJECT_CONTEXT}

Project Statistics

Historical Dataset
------------------
Rows: {data["historical_rows"]}

Columns:
{data["historical_columns"]}

Fear & Greed Dataset
--------------------
Rows: {data["fear_rows"]}

Columns:
{data["fear_columns"]}

Instructions

- Answer like a professional AI Financial Analyst.
- Use ONLY the supplied project information.
- Do not invent numbers.
- Keep the response concise and natural.

User Question:
{question}
"""

        return llm.invoke(prompt).content

    # ==================================================
    # PnL Statistics
    # ==================================================

    if any(word in q for word in ["pnl", "profit", "loss"]):

        data = pnl_statistics()

        prompt = f"""
{PROJECT_CONTEXT}

PnL Statistics

Mean Profit/Loss:
{data["mean"]}

Median:
{data["median"]}

Maximum Profit:
{data["max_profit"]}

Maximum Loss:
{data["max_loss"]}

Total Profit:
{data["total_profit"]}

Instructions

- Explain these statistics like a quantitative financial analyst.
- Highlight important observations.
- Mention risks if appropriate.

User Question:
{question}
"""

        return llm.invoke(prompt).content

    # ==================================================
    # Top Coins
    # ==================================================

    if any(word in q for word in ["coin", "token", "crypto"]):

        data = top_coins()

        prompt = f"""
{PROJECT_CONTEXT}

Trading Statistics

Top Traded Coins

{data}

Instructions

- Answer naturally.
- Do NOT generate Python code.
- Do NOT mention dictionaries.
- Give direct answers.

User Question:
{question}
"""

        return llm.invoke(prompt).content

    # ==================================================
    # Buy / Sell Ratio
    # ==================================================

    if "buy" in q or "sell" in q:

        data = buy_sell_ratio()

        prompt = f"""
{PROJECT_CONTEXT}

Trading Activity

Buy / Sell Ratio

{data}

Instructions

- Explain what this trading activity suggests.
- Mention whether buying or selling dominates.
- Keep the explanation concise.

User Question:
{question}
"""

        return llm.invoke(prompt).content

    # ==================================================
    # Fear & Greed
    # ==================================================

    if any(word in q for word in ["fear", "greed", "sentiment"]):

        data = sentiment_distribution()

        prompt = f"""
{PROJECT_CONTEXT}

Fear & Greed Distribution

{data}

Instructions

- Explain the sentiment distribution.
- Mention what it indicates about market psychology.
- Answer naturally.

User Question:
{question}
"""

        return llm.invoke(prompt).content

    # ==================================================
    # Predictions
    # ==================================================

    if "prediction" in q or "predict" in q:

        return latest_predictions()

    # ==================================================
    # Trading Signals
    # ==================================================

    if "signal" in q:

        return latest_signals()

    # ==================================================
    # General AI Questions
    # ==================================================

    prompt = f"""
{PROJECT_CONTEXT}

You are a helpful AI Financial Analyst.

If the user's question is unrelated to the project,
answer it normally using your general knowledge.

User Question:
{question}
"""

    return llm.invoke(prompt).content