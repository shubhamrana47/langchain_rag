import os 
import numpy as np
from dotenv import load_dotenv
from google import genai 

load_dotenv()



client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def get_embedding(text):
   response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text,
)
   return response.embeddings[0].values

text1 = "Artificial intelligence is a field of computer science "

text2 = "Machine learning  is a subfield of artificial intelligence."

text3 = "Deep learning is a subfield of machine learning ."


embedding1 = get_embedding(text1)
embedding2 = get_embedding(text2)
embedding3 = get_embedding(text3)

similarity_1_2 = np.dot(embedding1,embedding2) / (
    np.linalg.norm(embedding1) *
    np.linalg.norm(embedding2)
)


similarity_1_3 = np.dot(embedding1, embedding3) / (
    np.linalg.norm(embedding1) *
    np.linalg.norm(embedding3)
)
similarity_2_3 = np.dot(embedding2, embedding3) / (
    np.linalg.norm(embedding2) *
    np.linalg.norm(embedding3)
)


print("Text 1:", text1)
print("Text 2:", text2)
print("Text 3:", text3)

print()

print("Similarity between text 1 and text 2:", similarity_1_2)

print("Similarity between text 1 and text 3:", similarity_1_3)

print("Similarity between text 2 and text 3:", similarity_2_3)
