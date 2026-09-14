import os

import numpy as np
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


def cosine_similarity(vector_a, vector_b):

    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )



text = load_document("../../data/knowledge.txt")



chunks = chunk_text(text)


documents = []

for chunk in chunks:

    embedding = get_embedding(chunk)

    documents.append({
        "text": chunk,
        "embedding": embedding
    })


query = "What is deep learning?"


query_embedding = get_embedding(query)


results = []

for document in documents:

    score = cosine_similarity(
        query_embedding,
        document["embedding"]
    )

    results.append({
        "text": document["text"],
        "score": score
    })


results.sort(
    key=lambda x: x["score"],
    reverse=True
)


top_results = results[:2]


context = "\n".join(
    result["text"]
    for result in top_results
)


prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{query}
"""


response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)


print("\nFinal Answer:")
print(response.text)