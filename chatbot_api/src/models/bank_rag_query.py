from pydantic import BaseModel


class BankQueryInput(BaseModel):
    text: str


class BankQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]
