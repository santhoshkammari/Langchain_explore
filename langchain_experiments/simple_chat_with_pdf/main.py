from typing import List

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_experiments.simple_chat_with_pdf.const import PDF_PATH, TEXT_SPLIT_CHUNK_SIZE, TEXT_SPLIT_CHUNK_OVERLAP, \
    EMBEDDING_MODEL_NAME, VECTOR_DB_COLLECTION_NAME, LLM_MODEL_NAME

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_experiments.simple_chat_with_pdf.prompt_template import  RETRIEVER_PROMPT_TEMPLATE


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

def vector_store(chunks):
    filtered_chunks = []
    for chunk in chunks:
        filtered_chunks.append(Document(page_content=chunk.page_content))
    vector_db = Chroma.from_documents(
        documents=filtered_chunks,
        embedding=OllamaEmbeddings(
            model = EMBEDDING_MODEL_NAME,
            show_progress = True
        ),
        collection_name= VECTOR_DB_COLLECTION_NAME
    )
    return vector_db

def load_llm():
    __llm = ChatOllama(model = LLM_MODEL_NAME)
    return __llm

def load_hf_model():
    __llm = HuggingFacePipeline.from_model_id(
        model_id="openai-community/gpt2",
        task="text-generation",
        model_kwargs={
            "max_length": 250,
                        }
    )
    return __llm
def get_retriever(vector_db=None,llm = None):
    retriever_query_prompt = PromptTemplate(
        input_variables=["question"],
        template=RETRIEVER_PROMPT_TEMPLATE
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(),
        llm=llm,
        prompt=retriever_query_prompt
    )
    return retriever

if __name__ == '__main__':
    page_wise_document: List[Document] = read_pdf()
    chunks: List[Document] = create_chunks(page_wise_document)
    vector_db:Chroma = vector_store(chunks)
    llm = load_hf_model()
    retriever = get_retriever(vector_db,llm)

    template = '''
    Answer the question based only on the following context:
    {context}
    Question: {question}
    '''

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context":retriever,"question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_input = """
    Provide the value of 59 ?
    """
    response = chain.invoke(
        input=chain_input
    )
    print(response)






