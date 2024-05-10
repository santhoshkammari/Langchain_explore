from typing import List

import torch
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import GPT2Tokenizer

from langchain_experiments.simple_chat_with_pdf.const import PDF_PATH, TEXT_SPLIT_CHUNK_SIZE, TEXT_SPLIT_CHUNK_OVERLAP, \
    LLM_MODEL_NAME


def read_pdf():
    pdf_loader = UnstructuredPDFLoader(
        file_path=PDF_PATH,
        mode="paged"
    )
    data = pdf_loader.load()
    return data


def create_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= TEXT_SPLIT_CHUNK_SIZE,
        chunk_overlap = TEXT_SPLIT_CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def run_chain(retriever=None, llm=None):
    template = '''
        Answer the question based only on the following context:
        {context}
        Question: {question}
        '''

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
            {"context": retriever , "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    chain_input = """
        Extract the information for the key : 32B
        """
    print("Answer \n")
    response = chain.invoke(
        input=chain_input
    )
    return response

def load_hf_model():
    __llm = HuggingFacePipeline.from_model_id(
        model_id="openai-community/gpt2",
        task="text-generation",
        model_kwargs={
            "max_length": 250,
                        }
    )
    return __llm

def get_hugging_faceinference_embeddings():
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        model_name = "gpt2",
        api_key = "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"
    )
    return embeddings


def main():
    page_wise_document: List[Document] = read_pdf()
    chunks: List[Document] = create_chunks(page_wise_document)
    chunks = [Document(page_content=_.page_content) for _ in chunks]
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token  # You can choose another token if needed

    vectorstore = Chroma.from_documents(documents=chunks,
                                        embedding=get_hugging_faceinference_embeddings()
                                         )
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model=LLM_MODEL_NAME)
    res = run_chain(retriever=retriever,
                    llm=llm)
    print(res)


if __name__ == '__main__':
    main()

