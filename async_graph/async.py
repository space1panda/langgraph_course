import time
from dotenv import load_dotenv
load_dotenv()

import operator
from typing import Annotated, Any, TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    aggregate: Annotated[list, operator.add]


class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        time.sleep(1)
        print(f"Addding {self._value} to {state['aggregate']}")
        return {"aggregate": [self._value]}


builder = StateGraph(State)
builder.add_node('a', ReturnNodeValue("A"))
builder.add_edge(START, "a")
builder.add_node('b', ReturnNodeValue("B"))
builder.add_node('b2', ReturnNodeValue("B2"))
builder.add_node('c', ReturnNodeValue("C"))
builder.add_node('d', ReturnNodeValue("D"))

builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "b2")
builder.add_edge(["b2", "c"], "d")
builder.add_edge("d", END)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path='async.png')


if __name__ == '__main__':
    print("Hello async graph")
    graph.invoke({"aggregate": []}, {"configurable": {"thread_id": "food"}})
