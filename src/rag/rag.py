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


def get_embedding(text):
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


# User question
query =input("ask the question ?")


# Convert question into embedding
query_embedding = get_embedding(query)


# Search Chroma
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)


# Get retrieved chunks
retrieved_chunks = results["documents"][0]


print("\nRetrieved chunks:")

for chunk in retrieved_chunks:
    print("\n", chunk)


# Combine chunks into context
context = "\n".join(retrieved_chunks)


# Create prompt for Gemini
prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{query}
"""


# Ask Gemini to generate the answer
response = gemini_client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)


print("\nFinal Answer:")
print(response.text)