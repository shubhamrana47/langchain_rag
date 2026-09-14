import os 
from dotenv import load_dotenv
from google import genai 

load_dotenv()
api_key=os.getenv("gemini api key ")
client=genai.Client(api_key=api_key)

response=client.models.generate_content(
   model="gemini-3.6-flash",
    contents="what is rag in simple words "
)

print(response.text)