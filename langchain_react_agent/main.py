from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

search_tool = TavilySearchResults(search_depth="basic")


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current system time in the specified format"""
    return datetime.now().strftime(format)


tools = [search_tool, get_system_time]

agent = initialize_agent(
    tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True
)

agent.invoke(
    {
        "input": "When was SpaceX's last launch and how many days ago was it from this instant?"
    }
)
