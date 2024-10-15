from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (JsonOutputToolsParser,
                                                        PydanticToolsParser)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()


llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher. Current time {time}.
        
        1. {first_instruction}
        2. Reflect and critique your answer. Be severe to maximize the improvement.
        3. Recommend search queries to research information and improve your answer""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("answer the user's question above using the required format"),
    ]
).partial(time=lambda: datetime.now().isoformat())

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
