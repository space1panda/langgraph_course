from typing import Any, Dict, List

from corrective_rag.graph.chains.doc_grader import doc_grader
from graph.state import GraphState
from langchain.schema import Document


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents: List[Document] = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = doc_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
