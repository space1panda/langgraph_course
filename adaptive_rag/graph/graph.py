from dotenv import load_dotenv
from graph import consts
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from langgraph.graph import END, StateGraph
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from graph.chains.router import RouteQuery, question_router


load_dotenv()


def web_gate(state: GraphState) -> str:
    if state["web_search"]:
        return consts.WEBSEARCH
    return consts.GENERATE


def generation_gate(state: GraphState) -> str:
    print("--CHECK HALLUCINATIONS")
    question = state['question']
    documents = state['documents']
    generation = state['generation']

    score = hallucination_grader.invoke({
        "documents": documents, "generation": generation
    })

    if hallucination := score.score:
        print("--GROUNDED--")
        score = answer_grader.invoke({
            "question": question, "generation": generation
        })

        if answer:= score.binary_score:
            print("--RELEVANT--")
            return 'useful'
        else:
            return 'not useful'
    else:
        return 'not supported'


def route_question(state: GraphState) -> None:
    print("--ROUTE QUESTION--")
    question = state['question']
    source: RouteQuery = question_router.invoke({
        "question": question
    })
    if source.datasource == consts.WEBSEARCH:
        print("--ROUTE QUESTION TO WEBSEARCH")
        return consts.WEBSEARCH
    elif source.datasource == 'vectorstore':
        return consts.RETRIEVE


flow = StateGraph(GraphState)

# Add nodes

flow.add_node(consts.GENERATE, generate)
flow.add_node(consts.GRADE_DOCUMENTS, grade_documents)
flow.add_node(consts.RETRIEVE, retrieve)
flow.add_node(consts.WEBSEARCH, web_search)
# flow.set_entry_point(consts.RETRIEVE)
flow.set_conditional_entry_point(route_question, {consts.WEBSEARCH: consts.WEBSEARCH, consts.RETRIEVE: consts.RETRIEVE})

# Add edges

flow.add_edge(consts.RETRIEVE, consts.GRADE_DOCUMENTS)
flow.add_conditional_edges(
    consts.GRADE_DOCUMENTS,
    web_gate,
    {consts.WEBSEARCH: consts.WEBSEARCH, consts.GENERATE: consts.GENERATE},
)

flow.add_conditional_edges(
    consts.GENERATE, generation_gate, path_map={
        "not useful": consts.WEBSEARCH,
        "useful": END,
        "not supported": consts.GENERATE
        }
)
flow.add_edge(consts.WEBSEARCH, consts.GENERATE)
flow.add_edge(consts.GENERATE, END)
app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="corrective_rag_graph.png")
