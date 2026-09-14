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

# Chroma collection
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


# Create embeddings and store in Chroma
for i, chunk in enumerate(chunks):

    embedding = get_embedding(chunk)

    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embedding]
    )

    print(f"Stored chunk {i + 1}")

    # User question
query = "What is deep learning?"

# Create embedding for the question
query_embedding = get_embedding(query)

# Search Chroma
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\nRetrieved chunks:")

for document in results["documents"][0]:
    print("\n", document)