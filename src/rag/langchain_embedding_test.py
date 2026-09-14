import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


text = "Deep learning is a subfield of machine learning."

vector = embeddings.embed_query(text)


print("Embedding length:", len(vector))
print("First 5 values:", vector[:5])