from typing import List, Sequence
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
)
from langgraph.graph import END, MessageGraph, START
from chains import generate_chain, reflect_chain

from langchain_core.runnables.graph import MermaidDrawMethod


# define node names

REFLECT = "reflect"
GENERATE = "generate"


# define chains that should be run when entering the nodes


def generation_node(state: Sequence[BaseMessage]):
    """Generation node call"""
    return generate_chain.invoke({"messages": state})


def reflection_node(
    messages: Sequence[BaseMessage],
) -> List[BaseMessage]:
    """Reflection node call"""
    res = reflect_chain.invoke({"messages": messages})

    # ATTENTION: reflection node should mimic human-feedback
    return [HumanMessage(content=res.content)]


def main() -> MessageGraph:
    """Building simple Graph Agent with 2 nodes"""
    # Graph building

    builder = MessageGraph()

    # Define graph nodes
    builder.add_node(GENERATE, generation_node)
    builder.add_node(REFLECT, reflection_node)

    # Define where do we start. Which node should accept the user prompt?
    builder.set_entry_point(GENERATE)
    # builder.add_node(START, generation_node)

    # A callback for creating conditional edge of the graph. In this case - stop reflecting after lengths of state reaches 6 messages

    def should_continue(state: List[BaseMessage]):
        if len(state) > 6:
            return END
        return REFLECT

    # We connect GENERATE -> REFLECT through conditional edge

    builder.add_conditional_edges(GENERATE, should_continue)

    # Establish backward connection from reflect to generate. This way the generate callback function will get the human-mimic message from reflector

    builder.add_edge(REFLECT, GENERATE)

    # Compile graph

    graph = builder.compile()
    return graph


if __name__ == "__main__":
    print("Hello LangGraph")
    graph = main()
    graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
        output_file_path="out.png",
    )
    # inputs = HumanMessage(
    #     content="""Make this tweet better:
    # @LangChainAI
    # -newly Tool Calling feature is seriouslt underrated. After a long wait, it's here with function calling possibilities. Made a video covering their newest blog post"""
    # )
    # response = graph.invoke(inputs)
    # print(response[-1].content)
