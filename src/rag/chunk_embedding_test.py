import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def chunk_text(text):
    return text.split("\n\n")


def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


text = load_document("../../data/knowledge.txt")

chunks = chunk_text(text)


for i, chunk in enumerate(chunks):

    embedding = get_embedding(chunk)

    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)

    print("Embedding length:", len(embedding))