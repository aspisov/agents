from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

generation_system_prompt = """
You are a twitter techie influencer assistant tasked with writing excellent twitter posts.
Generate the best twitter post possible for the user's request.
If the user provides critique, respond with a revised version of your previous attempts.
"""

reflection_system_prompt = """
You are a viral twitter influencer grading a tweet. 
Generate critique and recommendations for the user's tweet.
Always provide detailed recommendations, including requests for length, virality, style, etc.
"""

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            generation_system_prompt,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            reflection_system_prompt,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini")

generation_chain = generation_prompt | llm

reflection_chain = reflection_prompt | llm
