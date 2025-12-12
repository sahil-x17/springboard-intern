import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_chunks(text, chunk_size=2000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    # Path to merged file from Task 1 
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    file_path = os.path.abspath(os.path.join(base_dir, "..", "task1", "merged_output.txt"))

    print("Loading merged_output.txt ...")
    full_text = load_text_file(file_path)

    print("Splitting text into chunks ...")
    chunks = create_chunks(full_text)

    print(f"Total chunks created: {len(chunks)}")


    output_path = os.path.join(base_dir, "chunks_output.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n{chunk}\n\n")

    print(f"Chunks saved to: {output_path}")
