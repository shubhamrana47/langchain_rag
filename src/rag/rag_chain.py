import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# Determine project paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "HISTORY-ENGLISH.pdf"
PERSIST_DIR = PROJECT_ROOT / "storage" / "chroma_history"

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()


# --------------------------------------------------
# 1. CREATE EMBEDDING MODEL
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# --------------------------------------------------
# 2. STORE CHUNKS IN CHROMA / LOAD EXISTING STORE
# --------------------------------------------------

vectorstore = Chroma(
    persist_directory=str(PERSIST_DIR),
    embedding_function=embeddings
)

# Check if vectorstore already has indexed chunks
doc_count = vectorstore._collection.count()
reindex = "--reindex" in sys.argv
if reindex:
    sys.argv.remove("--reindex")

if doc_count > 0 and not reindex:
    print(f"Loaded existing vector store with {doc_count} chunks from {PERSIST_DIR}")
else:
    # --------------------------------------------------
    # LOAD DOCUMENT & SPLIT INTO CHUNKS
    # --------------------------------------------------
    # loader = TextLoader(str(PROJECT_ROOT / "data" / "knowledge.txt"))
    print(f"Loading document from: {PDF_PATH}")
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    print("Number of pages:", len(documents))
    print("Number of chunks:", len(chunks))

    print(f"Indexing {len(chunks)} chunks into Chroma...")
    # Batch indexing to respect Gemini API free tier rate limits (100 embed requests/min)
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        print(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        # Retry with backoff if rate limit is reached
        for attempt in range(5):
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = 35 * (attempt + 1)
                    print(f"Rate limit reached. Waiting {wait_time}s before retrying batch {batch_num}...")
                    time.sleep(wait_time)
                else:
                    raise

        # Pause between batches if more remain to stay under free tier limit
        if i + batch_size < len(chunks):
            print("Waiting 35s to respect Gemini API rate limits...")
            time.sleep(35)

    print(f"Indexing complete! Total chunks stored: {vectorstore._collection.count()}")


# --------------------------------------------------
# 3. CREATE RETRIEVER
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


# --------------------------------------------------
# 4. CREATE PROMPT
# --------------------------------------------------

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""
)


# --------------------------------------------------
# 5. CREATE GEMINI LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# --------------------------------------------------
# 6. USER QUESTION
# --------------------------------------------------

if len(sys.argv) > 1:
    question = " ".join(sys.argv[1:])
    print(f"\nEnter the question: {question}")
else:
    try:
        question = input("\n enter the question:")
    except EOFError:
        question = "What is the subject of this book?"


# --------------------------------------------------
# 7. RETRIEVE RELEVANT DOCUMENTS
# --------------------------------------------------

results = retriever.invoke(question)


# --------------------------------------------------
# 8. CREATE CONTEXT
# --------------------------------------------------

context = "\n\n".join(
    document.page_content
    for document in results
)


# --------------------------------------------------
# 9. CREATE FINAL PROMPT
# --------------------------------------------------

final_prompt = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)


# --------------------------------------------------
# 10. SEND PROMPT TO GEMINI
# --------------------------------------------------

response = llm.invoke(final_prompt)


# --------------------------------------------------
# 11. PRINT ANSWER
# --------------------------------------------------

print("\nAnswer:")
answer = response.text if hasattr(response, "text") and response.text else response.content
print(answer)