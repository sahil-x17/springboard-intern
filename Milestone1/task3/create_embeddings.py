import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load chunks from Task 2 output file
def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()

    # Split using --- Chunk X ---
    chunks_raw = data.split("--- Chunk")
    chunks = []

    for item in chunks_raw:
        if item.strip():
            # Remove the header "X ---"
            parts = item.split("---", 1)
            if len(parts) == 2:
                text = parts[1].strip()
                chunks.append(text)

    return chunks


# Create embeddings for all chunks
def create_embeddings(chunks):
    print("Loading HuggingFace model: all-MiniLM-L6-v2 ...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Model loaded successfully!")

    output = []

    for i, text in enumerate(chunks):
        embedding = embedding_model.embed_query(text)  # Convert to vector

        output.append({
            "id": i,
            "text": text,
            "embedding": embedding
        })

    return output


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to chunk file from Task 2
    chunk_file = os.path.abspath(os.path.join(base_dir, "..", "task2", "chunks_output.txt"))

    print("Loading chunks from Task 2...")
    chunks = load_chunks(chunk_file)

    print(f"Total chunks found: {len(chunks)}")

    print("Creating embeddings ...")
    results = create_embeddings(chunks)

    # Save JSON output
    output_file = os.path.join(base_dir, "embeddings_output.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nEmbeddings saved successfully!")
    print(f"File: {output_file}")
