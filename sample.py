from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage




from pdf2image import convert_from_path
from pytesseract import image_to_string

from const import CHUNK_OVERLAP, CHUNK_SIZE, LLM_MODEL_NAME

from functools import lru_cache



def sample_chat_invoke():
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system',"hi act like a Doctor!"),
            ("user","{input}")
        ]
    )


    chain = prompt | llm | output_parser

    res = chain.invoke(
        {
            "input":"tell me a joke of medicine in 5 words!"
        }
    )

class CustomDocumentLoader():
    def __init__(self,pdf_path:str) -> None:
        self.pdf_path: str = pdf_path
        self.ocr_text: str = self.ocr_pdf()

    @lru_cache(maxsize=None)
    def ocr_pdf(self):
        print("OCR START")
        pages = convert_from_path(self.pdf_path)

        # Initialize empty list to store OCR results for each page
        ocr_results = []

        # Perform OCR on each page and append the result to ocr_results
        for page in pages:
            # Perform OCR using pytesseract
            text = image_to_string(page)

            # Append OCR result to ocr_results
            ocr_results.append(text)
        print("OCR DONE")
        return ocr_results
    
    @lru_cache(maxsize=None)
    def create_document(self):
        ocr_results = self.ocr_text
        document_lines = []
        for page_no,page in enumerate(ocr_results):
            # lines = page.split("\n")
            text_splitter = RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                chunk_size= CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False,
            )
            lines = text_splitter.split_text( text= page)
            for line_number,line in enumerate(lines):
                if not line: continue
                document_lines.append(Document(
                            page_content=line,
                            metadata={
                                "line_number": line_number+1, 
                                "source": self.pdf_path,
                                "page": page_no+1
                            })
                )
                                    

        return document_lines
def format_docs(docs: List[Document]):
    print("===================")
    print(docs)
    print("=================")
    return docs

def main():
    llm = Ollama(model = LLM_MODEL_NAME)
    output_parser = StrOutputParser()

    data = CustomDocumentLoader('langchain_experiments/only_amendment_case_001.pdf')
    documents = data.create_document()


    print('DOCUMENTS CREATION DONE')
    embeddings = OllamaEmbeddings(model = LLM_MODEL_NAME)
    vector = FAISS.from_documents(documents, embeddings)


    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    # resp = document_chain.invoke({
    #     "input": "how can langsmith help with testing?",
    #     "context": [Document(page_content="langsmith can let you visualize test results")]
    # })


    retriever = vector.as_retriever()
    retrieval_chain = {"context": retriever | format_docs,"input":RunnablePassthrough()}  | prompt | llm | output_parser
    response = retrieval_chain.invoke("Extract the value of key named 47B")

    print(response)
    exit()
    # print(response["context"])
    # print(response["answer"])

    # prompt = ChatPromptTemplate.from_messages([
    # MessagesPlaceholder(variable_name="chat_history"),
    # ("user", "{input}"),
    # ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    # ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    resp = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })

    print(resp)



def test():
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()
    print(docs)
    print(type(docs))
if __name__ == "__main__":
    # test()
    main()




