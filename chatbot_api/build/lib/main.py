from fastapi import FastAPI
from agents.bank_rag_agent import bank_rag_agent_executor
from models.bank_rag_query import BankQueryInput, BankQueryOutput
from utils.async_utils import async_retry

app = FastAPI(
    title="Bank Chatbot",
    description="Endpoints for a banking system graph RAG chatbot",
)


@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """
    Retry the agent if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """

    return await bank_rag_agent_executor.invoke({"input": query})


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/hospital-rag-agent")
async def ask_bank_agent(query: BankQueryInput) -> BankQueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    return query_response
