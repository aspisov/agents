import json

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import ToolMessage
from langgraph.graph import MessagesState

search_tool = TavilySearchResults(max_results=5)


def execute_tools(state: MessagesState):
    last_ai_message = state["messages"][-1]

    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []

    tool_messages = []

    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])

            search_results = {}

            for query in search_queries:
                search_results[query] = search_tool.invoke(query)

            tool_messages.append(
                ToolMessage(content=json.dumps(search_results), tool_call_id=call_id)
            )

    return {"messages": tool_messages}
