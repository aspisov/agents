import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_openai import ChatOpenAI

from langgraph_reflexion_agent.schema import AnswerQuestion

load_dotenv()

llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
)


pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

system_prompt = """
You are an expert AI researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries sepately** for researching improvements. Do not include them inside the reflection.
"""

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide me a detailed ~250 word answer."
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instruction = """
Revise your previous answer using the new information.
- You should use previous critique to add important information to your answer.
    - You must include numerial citations in your revised answer to ensure it can be verified.
    - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
        - [1] https://example.com
        - [2] https://example.com
- You should use the previous critique to remove superfluous information from your answer and make sure it is not more then 250 words.
"""

revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_instruction
) | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

response = first_responder_chain.invoke(
    {
        "messages": [
            HumanMessage(
                content="Write be a blog post on how to create a ReAct agent with Langgraph."
            )
        ]
    }
)
