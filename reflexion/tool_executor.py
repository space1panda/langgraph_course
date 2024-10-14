import json
from collections import defaultdict
from typing import List, Dict, Any
from dotenv import load_dotenv
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    AIMessage,
)

from schemas import AnswerQuestion, Reflection
from actor import parser


load_dotenv()


search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
tool_executor = ToolExecutor([tavily_tool])


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    tool_invocation = state[-1]
    parsed_tool_calls = parser.invoke(tool_invocation)

    ids = []
    tool_invocations = []

    for call in parsed_tool_calls:
        for query in call["args"]["search_queries"]:
            tool_invocations.append(
                ToolInvocation(
                    tool="tavily_search_results_json", tool_input=query
                )
            )
            ids.append(call["id"])

    outputs = tool_executor.batch(tool_invocations)
    output_map: Dict[str, Dict[Any, Any]] = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        output_map[id_][invocation.tool_input] = output

    tool_messages = []
    for id_, mapped_output in output_map.items():
        tool_messages.append(
            ToolMessage(content=json.dumps(mapped_output), tool_call_id=id_)
        )
    return tool_messages


if __name__ == "__main__":
    print("Tool Execution")

    human_message = HumanMessage(
        content="Write about influence of Finnish language on Tolkien's works"
    )

    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[
            "Examples of using Finnish in Quenya",
            "Tolkiens citations about Finnish language",
            "Tolkien's opinion on Kalevala",
        ],
        id="call_fST0rI2dsHAtm2mDwgi4Pcvu",
    )

    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.dict(),
                        "id": "call_fST0rI2dsHAtm2mDwgi4Pcvu",
                    }
                ],
            ),
        ]
    )

    print(raw_res)
