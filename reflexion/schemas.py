from typing import List

# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field


# Enforce structured output from agent


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question"""

    answer: str = Field(
        description="~250 words detailed answer to the question"
    )
    reflection: Reflection = Field(
        description="Your reflection on the original answer"
    )
    search_queries: List[str] = Field(
        description="1-3 search queries for research to improve the answer based on the current critique"
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question"""

    references: List[str] = Field(
        description="Citations motivating your updated answer"
    )
