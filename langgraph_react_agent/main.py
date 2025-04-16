import click
import yfinance as yf
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ------------------------------------------------------------------------------
# TOOLS
# ------------------------------------------------------------------------------


def get_stock_price(ticker: str) -> float:
    """Get the current price of a stock"""
    stock = yf.Ticker(ticker)
    return stock.info["previousClose"]


def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    return a / b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b


search = DuckDuckGoSearchRun()

tools = [multiply, add, divide, subtract, search, get_stock_price]

llm = ChatOpenAI(model="gpt-4o-mini")

llm_with_tools = llm.bind_tools(tools)

# ------------------------------------------------------------------------------
# NODE FUNCTIONS
# ------------------------------------------------------------------------------


def reasoner(state: MessagesState):
    system_message = SystemMessage(
        content="You are a helpful assistant tasked with using search and performing arithmetic on a set of inputs."
    )
    return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}


# ------------------------------------------------------------------------------
# GRAPH
# ------------------------------------------------------------------------------


class GraphState(MessagesState):
    query: str
    finance: str
    final_answer: str


workflow = StateGraph(GraphState)

workflow.add_node("reasoner", reasoner)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "reasoner")
workflow.add_conditional_edges(
    "reasoner",
    tools_condition,
)

workflow.add_edge("tools", "reasoner")

react_graph = workflow.compile()

try:
    react_graph.get_graph(xray=True).draw_mermaid_png(
        draw_method=MermaidDrawMethod.PYPPETEER,
        output_file_path="react_graph.png",
    )
except Exception as e:
    print(e)


@click.command()
@click.option("--query", type=str, help="The query to search for", default=None)
def main(query: str | None):
    if query is None:
        query = input("Enter a query: ")
    messages = [
        HumanMessage(content=query),
    ]
    result = react_graph.invoke({"messages": messages})
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()
