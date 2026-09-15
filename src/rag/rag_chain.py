import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================================
# PATH CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PERSIST_DIR = PROJECT_ROOT / "storage" / "chroma_history_huggingface"

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()


# ==========================================================
# 1. LOAD EMBEDDINGS & CONNECT TO STORED CHROMA DB
# ==========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect directly to the existing Chroma vector database on disk
vectorstore = Chroma(
    persist_directory=str(PERSIST_DIR),
    embedding_function=embeddings,
     collection_name="langchain"
)

doc_count = vectorstore._collection.count()
if doc_count == 0:
    print(f"Error: Vector DB at {PERSIST_DIR} is empty.")
    print("Please run 'python src/rag/ingest.py' to index your documents first.")
    sys.exit(1)

print(f"Connected to Chroma DB ({doc_count} chunks available).")


# ==========================================================
# 2. CREATE RETRIEVER
# ==========================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


# ==========================================================
# 3. CREATE PROMPT TEMPLATE
# ==========================================================

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question using only the information provided in the context.

If the answer is not present in the context, say:
"I don't have enough information in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
)


# ==========================================================
# 4. CREATE GEMINI LLM
# ==========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# ==========================================================
# 5. GET USER QUESTION
# ==========================================================

if len(sys.argv) > 1:
    question = " ".join(sys.argv[1:])
    print(f"\nEnter your question: {question}")
else:
    try:
        question = input("\nEnter your question: ")
    except EOFError:
        question = "What is the subject of this book?"


# ==========================================================
# 6. RETRIEVE RELEVANT DOCUMENTS FROM DB
# ==========================================================

results = retriever.invoke(question)


# ==========================================================
# 7. CREATE CONTEXT
# ==========================================================

context = "\n\n".join(
    document.page_content
    for document in results
)


# ==========================================================
# 8. CREATE FINAL PROMPT
# ==========================================================

final_prompt = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)


# ==========================================================
# 9. SEND PROMPT TO GEMINI
# ==========================================================

response = llm.invoke(final_prompt)


# ==========================================================
# 10. PRINT ANSWER
# ==========================================================

print("\nAnswer:")
answer = response.text if hasattr(response, "text") and response.text else response.content
print(answer)