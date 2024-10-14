from typing import List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

load_dotenv()

from tool_executor import execute_tools
from revisor import revisor
from actor import first_responder


MAX_ITERS = 2

builder = MessageGraph()
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)

builder.add_edge(start_key="draft", end_key="execute_tools")
builder.add_edge(start_key="execute_tools", end_key="revise")


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iters = count_tool_visits
    if num_iters > MAX_ITERS:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop)
builder.set_entry_point("draft")
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    print("Hello Reflexion")
    res = graph.invoke(
        "Write an article about connection between Tolkien and Finnish language"
    )
    print(res[-1].tool_calls[0]["args"]["answer"])
