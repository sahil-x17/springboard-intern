import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import random
load_dotenv()

# Load API key from environment variable
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("ERROR: Pinecone API key not found. Set PINECONE_API_KEY environment variable.")

# Initialize client
pc = Pinecone(api_key=api_key)

# Index name
index_name = "milestone-index"

# Create an Index
def create_index():
    print("Creating index:", index_name)

    pc.create_index(
        name=index_name,
        dimension=384,        
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Index created successfully!")


# Connect to Index
def connect_index():
    print("Connecting to index...")
    index = pc.Index(index_name)
    print("Connected!")
    return index


# Insert / Upsert (Create)
def insert_vectors(index):
    print("Upserting sample vectors...")

    # Random dummy embedding (384 dims)
    sample_vector = [random.random() for _ in range(384)]

    vectors = [
        {
            "id": "vec1",
            "values": sample_vector,
            "metadata": {"text": "This is a sample vector."}
        }
    ]

    index.upsert(vectors)
    print("Vector inserted.")


# Query (Read)

def query_vectors(index):
    print("Querying vector...")

    query_vector = [0.1] * 384   # simple query vector

    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    print("Query Results:")
    print(results)


# Update Vector (Update)
def update_vector(index):
    print("Updating vector metadata...")

    new_metadata = {"text": "Updated vector text"}

    index.update(
        id="vec1",
        set_metadata=new_metadata
    )

    print("Vector updated.")


# Delete Vector (Delete)
def delete_vector(index):
    print("Deleting vector...")

    index.delete(ids=["vec1"])

    print("Vector deleted.")


# Delete Index

def delete_index():
    print("Removing the index...")

    pc.delete_index(index_name)

    print("Index deleted successfully!")



# MAIN FLOW

if __name__ == "__main__":
    # 1. Create index
    create_index()

    # 2. Connect to index
    index = connect_index()

    # 3. CRUD Operations
    insert_vectors(index)
    query_vectors(index)
    update_vector(index)
    # delete_vector(index)

    # # 4. Delete index
    # delete_index()
