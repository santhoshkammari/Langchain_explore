from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
HuggingFaceHub()

## load using huggingface pipeline
# llm = HuggingFacePipeline(
#     model_id="gpt2",
#     task="text-generation",
#     # pipeline_kwargs={"max_new_tokens": 10},
# )
llm = HuggingFaceEndpoint(
        repo_id="openai-community/gpt2",
        task="text-generation",
        max_new_tokens = 20,
        model_kwargs = {}
    )
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models.huggingface import ChatHuggingFace

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What happens when an unstoppable force meets an immovable object?"),
]

chat_model = ChatHuggingFace(llm=llm)
print(chat_model.model_id)
exit('===')
# question = "What is electroencephalography?"
#
# print(hf.invoke(question))
# print(chain.invoke({"question": question}))

#
# QA_input = {
#     'question': 'Why is model conversion important?',
#     'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
# }
#
# template = ChatPromptTemplate.from_messages([
#         ("system","You're a helpful assistant"),
#         ("human","What happens when an unstoppable force meets an immovable object?")
#         ]
#     )
# prompt_value = template.invoke({})
# print(llm.invoke(prompt_value.to_messages()))
