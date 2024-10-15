from typing import Any, Dict

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from graph.state import GraphState

load_dotenv()

web_search_results = TavilySearchResults(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("--WEB-SEARCH--")
    question = state["question"]
    documents = state["documents"]
    tavily_results = web_search_results.invoke({"query": question})
    tavily_doc = "\n".join([t_res["content"] for t_res in tavily_results])
    web_res = Document(page_content=tavily_doc)

    if documents is None:
        documents = [web_res]
    else:
        documents.append(web_res)

    return {"question": question, "documents": documents}


if __name__ == "__main__":
    res = web_search(state={"question": "agent memory", "documents": None})
    print("ok")
