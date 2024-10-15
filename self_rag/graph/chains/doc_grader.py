from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")


class GradeDocuments(BaseModel):
    """Binary score for relevance check of the retrieved document"""

    score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_handler = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

doc_grader = grade_prompt | structured_llm_handler
