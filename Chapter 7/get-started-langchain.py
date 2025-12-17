from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings

from pymongo import MongoClient

import pprint
import os

# MongoDB connection info (with your info)
MONGO_URI = "YOUR_MONGODB_ATLAS_URI"
DB_NAME = "YOUR_DATABASE_NAME"
COLLECTION_NAME = "YOUR_COLLECTION_NAME"
INDEX_NAME = "YOUR_VECTOR_SEARCH_INDEX_NAME" # Atlas index name
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# MongoDB client and collection config
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# embedding model definition
embeddings = OpenAIEmbeddings()

# vector_store instance create
vector_store = MongoDBAtlasVectorSearch(
    collection=collection, 
    embedding=embeddings, 
    index_name=INDEX_NAME
)

query = "MongoDB Atlas security"
results = vector_store.similarity_search_with_score(
    query = query, k = 3
)
pprint.pprint(results)