import os
from typing import Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from src.chains.bank_faq_chain import faq_vector_chain
from src.chains.bank_cypher_chain import bank_cypher_chain
from src.tools.wait_times import (
    get_current_wait_times,
    get_most_available_branch,
)


BANK_AGENT_MODEL = os.getenv("BANK_AGENT_MODEL")

agent_chat_model = ChatOpenAI(
    model=BANK_AGENT_MODEL,
    temperature=0,
)


@tool
def explore_product_faqs(question: str) -> str:
    """
    Useful when you need to answer questions about product offerings,
    payment plans and interest rates. Not useful for answering objective questions that
    involve counting, percentages, aggregations, or listing facts.
    Use the entire prompt as input to the tool. For instance,
    if the prompt is "What are the different mortgage products",
    the input should be "What are the different products offered?".
    """

    return faq_vector_chain.invoke(question)


@tool
def explore_bank_database(question: str) -> str:
    """
    Useful for answering questions about customers,
    their mortgage/loan, payment schedule, fees, payments made by customer
    .Use the entire prompt as
    input to the tool. For instance, if the prompt is "What is the interest
    rate on a customer's mortgage?", the input should be "What is the interest
    rate on customer Jon Doe's loan?".
    """

    return bank_cypher_chain.invoke(question)


@tool
def get_branch_wait_time(branch: str) -> str:
    """
    Use when asked about current wait times
    at a specific branch. This tool can only get the current
    wait time at a branch and does not have any information about
    aggregate or historical wait times. Do not pass the word "branch"
    as input, only the branch name itself. For example, if the prompt
    is "What is the current wait time at Jordan Inc Branch?", the
    input should be "Jordan Inc".
    """

    return get_current_wait_times(branch)


@tool
def find_most_available_branch(tmp: Any) -> dict[str, float]:
    """
    Use when you need to find out which branch has the shortest
    wait time. This tool does not have any information about aggregate
    or historical wait times. This tool returns a dictionary with the
    branch name as the key and the wait time in minutes as the value.
    """

    return get_most_available_branch(tmp)

# @tool
# def get_customer(name: str)-> str:
#     """
#     Looks up for a customer by name
#     in the customer dataset and verifies the identity of the 
#     customer when the customer enters the name
#     """
#     return verify_customer(name)

agent_tools = [
    explore_product_faqs,
    explore_bank_database,
    get_branch_wait_time,
    find_most_available_branch,
]

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful chatbot for a bank designed to answer any queries
            about customer mortgage/loan details, customer payment schedule, fee related query and
            wait times and availability for appointment in a bank branch.
            """,
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent_llm_with_tools = agent_chat_model.bind_tools(agent_tools)

bank_rag_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | agent_prompt
    | agent_llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

bank_rag_agent_executor = AgentExecutor(
    agent=bank_rag_agent,
    tools=agent_tools,
    verbose=True,
    return_intermediate_steps=True,
)
