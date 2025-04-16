from chains import generation_chain, reflection_chain
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()


def generate_node(state: MessagesState):
    response = generation_chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def reflect_node(state: MessagesState):
    response = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content)]}


def should_continue_node(state: MessagesState):
    if len(state["messages"]) > 4:
        return END
    else:
        return "reflect"


builder = StateGraph(MessagesState)

builder.add_node("generate", generate_node)
builder.add_node("reflect", reflect_node)

builder.add_edge(START, "generate")
builder.add_conditional_edges(
    "generate",
    should_continue_node,
)
builder.add_edge("reflect", "generate")

graph = builder.compile()

graph.get_graph().draw_mermaid_png(
    output_file_path="langgraph_reflection_agent/diagram.png"
)

result = graph.invoke(
    {"messages": [HumanMessage(content="AI Agents are the future of risk management.")]}
)
for message in result["messages"]:
    message.pretty_print()
