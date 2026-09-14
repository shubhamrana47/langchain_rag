from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os 
load_dotenv()


# 1. Load document
loader = TextLoader("../../data/knowledge.txt")
documents = loader.load()


# 2. Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(documents)


# 3. Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
     google_api_key=os.getenv("GEMINI_API_KEY")
)


# 4. Create Chroma vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="../../storage/chroma_langchain",
)


# 5. Create Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


# 6. Ask a question
query = "What is deep learning?"


# 7. Retrieve relevant documents
results = retriever.invoke(query)


# 8. Print results
print("\nRetrieved documents:")

for i, result in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(result.page_content)