from pprint import pprint
from typing import cast

from graph.chains.doc_grader import GradeDocuments, doc_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import (GradeHallucinations,
                                               hallucination_grader)
from ingestion import retriever

DOC_GRADER_QUESTION_YES = "agent memory"
DOC_GRADER_QUESTION_NO = "african kitchen"


def test_doc_grader_yes() -> None:
    docs = retriever.invoke(DOC_GRADER_QUESTION_YES)
    doc_txt = docs[0].page_content
    inputs = {"question": DOC_GRADER_QUESTION_YES, "document": doc_txt}
    res = doc_grader.invoke(inputs)
    res = cast(GradeDocuments, res)

    assert res.score == "yes"


def test_doc_grader_no() -> None:
    docs = retriever.invoke(DOC_GRADER_QUESTION_NO)
    doc_txt = docs[0].page_content
    res = doc_grader.invoke({"question": DOC_GRADER_QUESTION_NO, "document": doc_txt})
    res = cast(GradeDocuments, res)

    assert res.score == "no"


def test_generation_chain() -> None:
    docs = retriever.invoke(DOC_GRADER_QUESTION_YES)
    generation = generation_chain.invoke(
        {"context": docs, "question": DOC_GRADER_QUESTION_YES}
    )
    pprint(generation)


def test_hallucination_grader_yes() -> None:
    docs = retriever.invoke(DOC_GRADER_QUESTION_YES)
    generation = generation_chain.invoke(
        {"context": docs, "question": DOC_GRADER_QUESTION_YES}
    )
    res: GradeHallucinations = hallucination_grader.invoke({
        "documents": docs,
        "generation": generation
    })
    assert res.score


def test_hallucination_grader_no() -> None:
    docs = retriever.invoke(DOC_GRADER_QUESTION_YES)
    res: GradeHallucinations = hallucination_grader.invoke({
        "documents": docs,
        "generation": "You have to break some eggs to make pizza"
    })
    assert not res.score
