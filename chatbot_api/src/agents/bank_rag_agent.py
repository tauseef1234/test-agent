import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field

# --- Import your existing chains and NEW verification chain ---
from src.chains.bank_faq_chain import faq_vector_chain
from src.chains.bank_cypher_chain import bank_cypher_chain
from src.chains.customer_verification_chain import customer_verification_chain
from src.tools.wait_times import (
    get_current_wait_times,
    get_most_available_branch,
)

# --- Model Definition (Unchanged) ---
BANK_AGENT_MODEL = os.getenv("BANK_AGENT_MODEL")
agent_chat_model = ChatOpenAI(
    model=BANK_AGENT_MODEL,
    temperature=0,
)

# --- Define the NEW Verification Tool ---

class CustomerVerificationInput(BaseModel):
    first_name: str = Field(description="The customer's first name.")
    last_name: str = Field(description="The customer's last name.")
    zip_code: str = Field(description="The customer's zip code.")
    phone_number: str = Field(description="The customer's 10-digit phone number.")

@tool(args_schema=CustomerVerificationInput)
def verify_customer(first_name: str, last_name: str, zip_code: str, phone_number: str) -> str:
    """
    Use this tool to verify a customer's identity using their first name, last name, 
    zip code, and phone number. Only use this tool after you have collected all four 
    pieces of information from the user.
    """
    return customer_verification_chain.invoke({
        "first_name": first_name,
        "last_name": last_name,
        "zip_code": zip_code,
        "phone_number": phone_number,
    })

# --- Define Existing Tools (Unchanged) ---
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
    This tool takes no input.
    """
    return get_most_available_branch(tmp)


# --- Update Agent Tools List ---
agent_tools = [
    verify_customer, 
    explore_product_faqs,
    explore_bank_database,
    get_branch_wait_time,
    find_most_available_branch,
]

# --- CRITICAL: Update the System Prompt ---
agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful and secure bank chatbot. Your primary goal is to assist customers with their accounts.

**Verification Protocol:**
1.  You MUST start every new conversation by stating that you need to verify the user's identity.
2.  Politely ask for the user's first name, last name, zip code, and phone number, one at a time.
3.  Once you have collected all four pieces of information, you MUST use the `verify_customer` tool.
4.  If verification is successful, the tool will return the Customer Name and ID. Acknowledge this to the user (e.g., "Thank you, John. You are verified."). From this point on, you can answer their specific questions.
5.  When using the `explore_bank_database` tool, you MUST include the customer's name or ID in the question you pass to it.
6.  If verification fails, inform the user and ask them to provide the information again. DO NOT attempt to use any other tools.

**General Rules:**
- Do not attempt to answer account-specific questions until verification is successful.
- Be polite and clear in your instructions.
"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# --- Agent Definition and Executor ---
agent_llm_with_tools = agent_chat_model.bind_tools(agent_tools)

bank_rag_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x.get("intermediate_steps", [])
        ),
        "chat_history": lambda x: x.get("chat_history", []),
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
    handle_parsing_errors=True,
)