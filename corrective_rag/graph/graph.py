from dotenv import load_dotenv
from graph import consts
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from langgraph.graph import END, StateGraph


load_dotenv()


def web_gate(state: GraphState) -> str:
    if state["web_search"]:
        return consts.WEBSEARCH
    return consts.GENERATE


flow = StateGraph(GraphState)

# Add nodes

flow.add_node(consts.GENERATE, generate)
flow.add_node(consts.GRADE_DOCUMENTS, grade_documents)
flow.add_node(consts.RETRIEVE, retrieve)
flow.add_node(consts.WEBSEARCH, web_search)
flow.set_entry_point(consts.RETRIEVE)

# Add edges

flow.add_edge(consts.RETRIEVE, consts.GRADE_DOCUMENTS)
flow.add_conditional_edges(
    consts.GRADE_DOCUMENTS,
    web_gate,
    {consts.WEBSEARCH: consts.WEBSEARCH, consts.GENERATE: consts.GENERATE},
)
flow.add_edge(consts.WEBSEARCH, consts.GENERATE)
flow.add_edge(consts.GENERATE, END)
app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="corrective_rag_graph.png")
