import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ ERROR: OPENAI_API_KEY not found in .env file")


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Ask a random question
response = llm.invoke("Explain recursion in simple words with an example.")

print("🔹 GPT-3.5 Turbo Response:\n")
print(response.content)
