import datetime
from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


load_dotenv()

# MessagePLaceholder - object for keeping space for new messages

actor_prompt_template = ChatMessagePromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher. Current time {time}.
        
        1. {first_instruction}
        2. Reflect and critique your answer. Be severe to maximize the improvement.
        3. Recommend search queries to research information and improve your answer""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "answer the user's question above using the required format"
        ),
    ]
).partial(
    time=lambda: datetime.datetime().now().isoformat()
)
