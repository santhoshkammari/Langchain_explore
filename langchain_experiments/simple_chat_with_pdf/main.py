from langchain_community.document_loaders import UnstructuredPDFLoader

from langchain_experiments.simple_chat_with_pdf.const import PDF_PATH, TEXT_SPLIT_CHUNK_SIZE, TEXT_SPLIT_CHUNK_OVERLAP, \
    EMBEDDING_MODEL_NAME, VECTOR_DB_COLLECTION_NAME

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata


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
    # filtered_chunks = [filter_complex_metadata(c) for c in chunks]
    print('===========')
    print(type(chunks))
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(
            model = EMBEDDING_MODEL_NAME,
            show_progress = True
        ),
        collection_name= VECTOR_DB_COLLECTION_NAME
    )
    return vector_db
if __name__ == '__main__':
    page_wise_document = read_pdf()
    chunks = create_chunks(page_wise_document)
    vector_store(chunks)
