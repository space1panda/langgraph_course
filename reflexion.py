"""https://arxiv.org/pdf/2303.11366 """

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
