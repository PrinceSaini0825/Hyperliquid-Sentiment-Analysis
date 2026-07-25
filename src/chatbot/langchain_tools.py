from langchain.tools import Tool

from src.chatbot.tools import (
    dataset_summary,
    pnl_statistics,
    top_coins,
    buy_sell_ratio,
    sentiment_distribution,
    latest_predictions,
    latest_signals,
)

tools = [

    Tool(
        name="Dataset Summary",
        func=lambda _: str(dataset_summary()),
        description="Answer questions about datasets, rows, columns and project data."
    ),

    Tool(
        name="PnL Statistics",
        func=lambda _: str(pnl_statistics()),
        description="Answer questions about profit, loss, pnl, returns and statistics."
    ),

    Tool(
        name="Top Coins",
        func=lambda _: str(top_coins()),
        description="Answer questions about traded coins and cryptocurrencies."
    ),

    Tool(
        name="Buy Sell Ratio",
        func=lambda _: str(buy_sell_ratio()),
        description="Answer questions about BUY SELL ratio and trading activity."
    ),

    Tool(
        name="Fear Greed",
        func=lambda _: str(sentiment_distribution()),
        description="Answer questions about Fear & Greed sentiment."
    ),

    Tool(
        name="Predictions",
        func=lambda _: str(latest_predictions()),
        description="Return latest model predictions."
    ),

    Tool(
        name="Trading Signals",
        func=lambda _: str(latest_signals()),
        description="Return latest trading signals."
    ),

]