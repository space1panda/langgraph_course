from typing import List, TypedDict

from langchain.schema import Document


class GraphState(TypedDict):
    """Descrition of graph state for RAG pipeline

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of relevant documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[Document]
