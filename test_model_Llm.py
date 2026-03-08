from langchain_ollama import ChatOllama
from src.config import LLM_MODEL
llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
resp = llm.invoke("bạn là ai?")
print("RAW:", resp)