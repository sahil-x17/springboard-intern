import os
import docx2txt
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

current_dir = os.path.dirname(os.path.abspath(__file__))

files = [
    "the_constitution_of_india.pdf",
    "legal document.txt",
    "THE INDIAN PENAL CODE.docx",
    "India Code_ Section Details"  
]

all_text = ""

for file in files:
    file_path = os.path.join(current_dir, file)

    print(f"Extracting: {file}")

    # PDF
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for d in docs:
            all_text += d.page_content + "\n\n"

    # TXT
    elif file.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            all_text += d.page_content + "\n\n"

    # DOCX (using docx2txt)
    elif file.endswith(".docx"):
        text = docx2txt.process(file_path)
        all_text += text + "\n\n"

    # HTML (using BeautifulSoup)
    elif file.endswith(".html") or file.endswith(".htm"):
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            soup = BeautifulSoup(html_content, "lxml")
            text = soup.get_text(separator="\n")
            all_text += text + "\n\n"

    else:
        print(f"Skipping unsupported file: {file}")

# Save outside
output_file = os.path.abspath(os.path.join(current_dir, "..", "merged_output.txt"))

with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_text)

print("\nExtraction completed successfully!")
print(f"File saved at: {output_file}")
