from typing import Any, Dict

from ..chains.generation import generation_chain


def generate(state) -> Dict[str, Any]:
    print("--GENERATE--")
    question = state["question"]
    documents = state["documents"]
    res = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": res}
