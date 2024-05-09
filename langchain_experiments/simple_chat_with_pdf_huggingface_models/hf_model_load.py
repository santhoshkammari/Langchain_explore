from contextlib import redirect_stderr

from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from transformers import AutoModelForQuestionAnswering
from dotenv import  load_dotenv
load_dotenv()

def load_hf():
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': 'Why is model conversion important?',
        'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    }
    res = nlp(QA_input)

    # b) Load model & tokenizer
    # model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    return res


def langchain_hf_load():
    question = "Who won the FIFA World Cup in the year 1994? "
    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    hf = HuggingFaceEndpoint(
        repo_id="openai-community/gpt2",
        task="text-generation",
        max_new_tokens = 20,
        model_kwargs = {}
    )
    chat_model = ChatHuggingFace(
        llm= hf
    )

    template = ChatPromptTemplate.from_messages([
        ("system","You're a helpful assistant"),
        ("human","What happens when an unstoppable force meets an immovable object?")
        ]
    )
    prompt_value = template.invoke({})
    print(chat_model.invoke(prompt_value.to_messages()))

    response = "done"
    return response


def testing():
    from langchain.llms.huggingface_pipeline import HuggingFacePipeline

    hf = HuggingFacePipeline.from_model_id(
        model_id="openai-community/gpt2",
        task="text-generation",
        # model_kwargs={
        #     "max_length": 250,
        #                 }
    )

    from langchain.prompts import PromptTemplate

    template = """Question: {question}"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | hf

    question = "What is electroencephalography?"

    print('==================')
    # print(chain.stream({"question": question}))
    for text in chain.stream({"question":question}):
        print(text, end="|", flush=True)

if __name__ == '__main__':
    # hf = load_hf_llm()
    # response = hf.invoke(
    #     input="hai"
    # )
    # print(response)
    print(testing())