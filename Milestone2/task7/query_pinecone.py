import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "task6-index"


# --------------------------------------------------
# CONNECT TO PINECONE
# --------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# CONNECT TO EXISTING INDEX (NEW WAY)
# --------------------------------------------------
vector_store = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embedding
)


# --------------------------------------------------
# QUERY
# --------------------------------------------------
query = input("🔍 Enter your query: ")

results = vector_store.similarity_search(query, k=4)


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
print("\nTop matching documents:\n")

for i, doc in enumerate(results, start=1):
    print(f"Result {i}:")
    print(doc.page_content[:500])
    print("-" * 80)
