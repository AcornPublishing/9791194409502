from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# according to Atlas Vector Search, you need to get other import
# ex: from langchain_community.vectorstores import MongoDBAtlasVectorSearch 
import pprint 

# please check import (refer to import)
# from langchain_openai import OpenAIEmbeddings 
# from pymongo import MongoClient 

# MongoDB connection info (with your info)
MONGO_URI = "YOUR_MONGODB_ATLAS_URI"
DB_NAME = "YOUR_DATABASE_NAME"
COLLECTION_NAME = "YOUR_COLLECTION_NAME"
INDEX_NAME = "YOUR_VECTOR_SEARCH_INDEX_NAME" # Atlas index name

# MongoDB client and collection config
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# embedding model definition
embeddings = OpenAIEmbeddings()

# vector_store instance create (assumption: docu is already embedding in collection)
vector_store = MongoDBAtlasVectorSearch(
    collection=collection, 
    embedding=embeddings, 
    index_name=INDEX_NAME
)

# Instantiate Atlas Vector Search as a retriever
retriever = vector_store.as_retriever(
  search_type = "similarity",
  search_kwargs = { "k": 3 }
)

# Define a prompt template
template = """
Use the following pieces of context to answer the question at the end.If
you don't know the answer, just say that you don't know, don't try to make
up an answer.
{context}

Question: {question}
"""
custom_rag_prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI()
def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

# Construct a chain to answer questions on your data
rag_chain = (
{ "context": retriever | format_docs, "question": RunnablePassthrough()}
  | custom_rag_prompt
  | llm
  | StrOutputParser()
)

# Prompt the chain
question = "How can I secure my MongoDB Atlas cluster?"
answer = rag_chain.invoke(question)
print(°ÏQuestion: °Ï + question)
print(°ÏAnswer: °Ï + answer)

# Return source documents
documents = retriever.get_relevant_documents(question)
print(°Ï\nSource documents:°Ì)
pprint.pprint(documents)