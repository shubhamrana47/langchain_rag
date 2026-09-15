from pathlib import Path
from dotenv import load_dotenv
import chromadb

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "HISTORY-ENGLISH.pdf"
PERSIST_DIR = PROJECT_ROOT / "storage" / "chroma_history_huggingface"


def ingest():
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()

    print(f"Loading document from: {PDF_PATH}")
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()
    print(f"Number of pages loaded: {len(documents)}")

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Number of chunks created: {len(chunks)}")

    print("Initializing HuggingFace embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Clear existing collection to avoid duplicate chunks
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    try:
        client.delete_collection("langchain")
        print("Cleared previous collection from database.")
    except Exception:
        pass

    print(f"Storing chunks and embeddings into Chroma DB at: {PERSIST_DIR}...")
    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name="langchain"
    )
    vectorstore.add_documents(chunks)
    print(f"Ingestion complete! Total clean chunks stored in DB: {vectorstore._collection.count()}")


if __name__ == "__main__":
    ingest()
