import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Dict, List, Literal, Optional

load_dotenv()

client = OpenAI()

ROUTER_AGENT_SYSTEM_PROMPT = open("D:/askello/src/agents/prompts/router_agent_prompt.txt", "r", encoding="utf-8").read()

class RouterResponse(BaseModel):
    routed_direction: Literal["General", "RAG"]
    message: Optional[str]

def router_agent(query: str) -> RouterResponse:
    """
    This router takes the agent query and determines if it needs to be routed to General to answer Generally or routed to RAG. 
    If it is routed to General then also provide a message that is to be sent. 

    Parameters:
        - query (str) -> Query you wanna search

    Returns:
        - routed_direction (Dict) -> If General or RAG. If general also the message that is to be sent back
    """

    user_prompt = f"For the given user query: {query}. Route the accordingly"

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system", 
                "content": ROUTER_AGENT_SYSTEM_PROMPT
            }, 
            {
                "role": "user", 
                "content": user_prompt
            },
        ], 
        text_format = RouterResponse
    )

    return response.output_parsed