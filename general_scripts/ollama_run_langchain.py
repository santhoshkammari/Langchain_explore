from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import Ollama

llm = Ollama(model="gpt2")

print(llm.invoke("tell me about partial functions in python"))