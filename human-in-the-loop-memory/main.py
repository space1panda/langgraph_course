from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    input: str
    user_feedback: str


def step_1(state: State) -> None:
    print("--State 1--")


def human_feedback(state: State) -> None:
    print("--Human Feedback--")


def step_3(state: State) -> None:
    print("--State 3--")


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

# memory = MemorySaver()
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)


graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])

graph.get_graph().draw_mermaid_png(output_file_path='graph.png')


if __name__ == '__main__':
    thread = {"configurable": {"thread_id": "100"}}
    # initial_input = {"input": "hello world"}

    # for event in graph.stream(initial_input, thread, stream_mode="values"):
    #     print(event)
    
    user_input = input()
    graph.update_state(thread, {"user_feedback": user_input}, as_node="human_feedback")
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
