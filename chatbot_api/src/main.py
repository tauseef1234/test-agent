import uuid
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Import the agent executor and message types
from src.agents.bank_rag_agent import bank_rag_agent_executor
from langchain_core.messages import AIMessage, HumanMessage

# Import your async retry utility if you have one, otherwise remove this line
# from src.utils.async_utils import async_retry

app = FastAPI(
    title="Retail Bank Chatbot",
    description="Endpoints for a banking system graph RAG chatbot with session management",
)

# --- Pydantic Models for API ---
class BankQueryInput(BaseModel):
    text: str
    session_id: Optional[str] = Field(
        None,
        description="A unique identifier for the conversation session. If not provided, a new session will be started.",
    )

class BankQueryOutput(BaseModel):
    output: str
    intermediate_steps: List[str]
    session_id: str

# --- In-Memory Session Storage ---
# WARNING: This is for demonstration purposes only and will not work
# with multiple server workers or in a stateless environment.
# For production, use a persistent, shared store like Redis or a database.
SESSIONS: Dict[str, List[Any]] = {}


# The async_retry decorator is optional. If you don't have this file, 
# you can remove the decorator from the function below.
# @async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str, chat_history: list):
    """
    Invokes the agent with a given query and its conversation history.
    """
    return await bank_rag_agent_executor.ainvoke(
        {"input": query, "chat_history": chat_history}
    )

# --- API Endpoints ---
@app.get("/")
async def get_status():
    return {"status": "running"}

@app.post("/bank-rag-agent", response_model=BankQueryOutput)
async def ask_bank_agent(query: BankQueryInput) -> BankQueryOutput:
    """
    Handles a user's query to the bank agent with session state.
    """
    # 1. Get or create a session ID
    session_id = query.session_id or str(uuid.uuid4())
    
    # 2. Retrieve the conversation history for this session
    chat_history = SESSIONS.get(session_id, [])

    # 3. Invoke the agent with the user's query and the session's history
    query_response = await invoke_agent_with_retry(query.text, chat_history)

    # 4. Update the session history with the new turn
    chat_history.append(HumanMessage(content=query.text))
    chat_history.append(AIMessage(content=query_response["output"]))
    SESSIONS[session_id] = chat_history

    # 5. Format and return the response
    intermediate_steps_str = [
        str(s) for s in query_response.get("intermediate_steps", [])
    ]

    return BankQueryOutput(
        output=query_response["output"],
        intermediate_steps=intermediate_steps_str,
        session_id=session_id,
    )