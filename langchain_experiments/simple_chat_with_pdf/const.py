PDF_PATH: str = "/home/ntlpt59/MAIN/LLM/Langchain_explore/langchain_experiments/simple_chat_with_pdf/only_amendment_case_001.pdf"

#text splitting
TEXT_SPLIT_CHUNK_SIZE: int = 100
TEXT_SPLIT_CHUNK_OVERLAP: int = 50

#vectorDB parameters
EMBEDDING_MODEL_NAME: str = "nomic-embed-text"
VECTOR_DB_COLLECTION_NAME: str = "local-rag"

#llm parameters
LLM_MODEL_NAME: str = "tinyllama"