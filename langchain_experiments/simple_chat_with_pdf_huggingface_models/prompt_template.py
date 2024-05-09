RETRIEVER_PROMPT_TEMPLATE: str = '''
You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retreive relevant documents
from the vector database.
By generating mulitple perspectives on the user question , your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by new lines
Origina question : {question}
'''

