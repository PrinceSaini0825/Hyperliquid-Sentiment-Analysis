from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain_ollama import ChatOllama

from src.chatbot.langchain_tools import tools


llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0,
)


financial_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)