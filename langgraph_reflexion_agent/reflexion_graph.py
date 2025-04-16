from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import END, MessagesState, StateGraph

from langgraph_reflexion_agent.chains import first_responder_chain, revisor_chain
from langgraph_reflexion_agent.execute_tools import execute_tools

MAX_ITERATIONS = 2

builder = StateGraph(MessagesState)


def first_responder_node(state: MessagesState):
    result = first_responder_chain.invoke({"messages": state["messages"]})
    return {"messages": result}


def revisor_node(state: MessagesState):
    result = revisor_chain.invoke({"messages": state["messages"]})
    return {"messages": result}


def should_continue(state: MessagesState):
    count_tool_calls = sum(isinstance(m, ToolMessage) for m in state["messages"])
    if count_tool_calls > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_node("draft", first_responder_node)
builder.add_node("revisor", revisor_node)
builder.add_node("execute_tools", execute_tools)

builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revisor")
builder.add_conditional_edges(
    "revisor",
    should_continue,
)
builder.set_entry_point("draft")

graph = builder.compile()

try:
    graph.get_graph().draw_mermaid_png(
        output_file_path="langgraph_reflexion_agent/diagram.png",
        draw_method=MermaidDrawMethod.PYPPETEER,
    )
except Exception as e:
    print(e)

response = graph.invoke(
    {
        "messages": [
            HumanMessage(content="Write about creating a ReAct agent with Langgraph."),
        ]
    }
)

for message in response["messages"]:
    message.pretty_print()
