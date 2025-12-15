import os
import uuid
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import docx2txt
from bs4 import BeautifulSoup


# --------------------------------------------------
# ENV + CLIENT SETUP
# --------------------------------------------------
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "task6-index"


# --------------------------------------------------
# CREATE / CONNECT INDEX
# --------------------------------------------------
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


# --------------------------------------------------
# EMBEDDING + SPLITTER
# --------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)


# --------------------------------------------------
# FILE LOADERS
# --------------------------------------------------
def load_file(file_path):
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()

    elif file_path.endswith(".txt"):
        return TextLoader(file_path, encoding="utf-8").load()

    elif file_path.endswith(".docx"):
        text = docx2txt.process(file_path)
        return [{"page_content": text}]

    elif file_path.endswith(".html"):
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "lxml")
            return [{"page_content": soup.get_text()}]

    else:
        raise ValueError("Unsupported file type")


# --------------------------------------------------
# MAIN INGESTION PIPELINE
# --------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

for file in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, file)

    try:
        print(f"\nProcessing file: {file}")

        documents = load_file(file_path)

        chunks = []
        for doc in documents:
            chunks.extend(
                text_splitter.split_text(doc["page_content"])
            )

        print(f"🔹 Chunks created: {len(chunks)}")

        vectors = []
        for chunk in chunks:
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding_model.embed_query(chunk),
                "metadata": {
                    "source_file": file,
                    "text": chunk
                }
            })

        index.upsert(vectors=vectors)
        print(f"Inserted into Pinecone: {file}")

    except Exception as e:
        print(f"Failed processing {file}: {e}")
