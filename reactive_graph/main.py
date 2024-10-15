from dotenv import load_dotenv

load_dotenv()

from langchain_core.agents import AgentFinish
from langgraph.graph import END, StateGraph
from nodes import execute_tools, run_agent_reasoning_engine
from state import AgentState

AGENT_REASON = "agent_reason"
ACT = "act"


def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else:
        return ACT


# Define graph with custom schema for react pipeline

flow = StateGraph(AgentState)

# Nodes

flow.add_node(AGENT_REASON, run_agent_reasoning_engine)
flow.add_node(ACT, execute_tools)

flow.set_entry_point(AGENT_REASON)

# Edges

flow.add_conditional_edges(AGENT_REASON, should_continue)

flow.add_edge(ACT, AGENT_REASON)
app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")


if __name__ == "__main__":
    print("Hello React Graph")
    res = app.invoke(
        input={"input": "What is the weather in SF in Celcius? Write it and triple it"}
    )

    print(res["agent_outcome"].return_values["output"])
