import os

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Gemini client
gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Chroma client
chroma_client = chromadb.PersistentClient(
    path="../../storage/chroma"
)

collection = chroma_client.get_or_create_collection(
    name="knowledge"
)


def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def chunk_text(text):
    return text.split("\n\n")


def get_embedding(text):
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


# Load document
text = load_document("../../data/knowledge.txt")

# Create chunks
chunks = chunk_text(text)

print("Number of chunks:", len(chunks))

# Create embeddings and store them
ids = []
embeddings = []

for i, chunk in enumerate(chunks):

    embedding = get_embedding(chunk)

    ids.append(str(i))
    embeddings.append(embedding)


collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)

print(f"Indexed {len(chunks)} chunks into Chroma.")