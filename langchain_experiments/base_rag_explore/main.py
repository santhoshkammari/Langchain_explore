from typing import List

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_experiments.simple_chat_with_pdf.const import PDF_PATH, TEXT_SPLIT_CHUNK_SIZE, TEXT_SPLIT_CHUNK_OVERLAP


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


if __name__ == '__main__':
    page_wise_document: List[Document] = read_pdf()
    chunks: List[Document] = create_chunks(page_wise_document)
    print(chunks)