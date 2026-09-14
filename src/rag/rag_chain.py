import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()


# --------------------------------------------------
# 1. LOAD DOCUMENT
# --------------------------------------------------

# loader = TextLoader("../../data/knowledge.txt")
loader=PyPDFLoader("../../data/HISTORY-ENGLISH.pdf")
documents = loader.load()


# --------------------------------------------------
# 2. SPLIT DOCUMENT INTO CHUNKS
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Number of pages:", len(documents))
print("Number of chunks:", len(chunks))

# --------------------------------------------------
# 3. CREATE EMBEDDING MODEL
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# --------------------------------------------------
# 4. STORE CHUNKS IN CHROMA
# --------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="../../storage/chroma_history"
)


# --------------------------------------------------
# 5. CREATE RETRIEVER
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


# --------------------------------------------------
# 6. CREATE PROMPT
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
# 7. CREATE GEMINI LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# --------------------------------------------------
# 8. USER QUESTION
# --------------------------------------------------

question = input("\n enter the question:")


# --------------------------------------------------
# 9. RETRIEVE RELEVANT DOCUMENTS
# --------------------------------------------------

results = retriever.invoke(question)


# --------------------------------------------------
# 10. CREATE CONTEXT
# --------------------------------------------------

context = "\n\n".join(
    document.page_content
    for document in results
)


# --------------------------------------------------
# 11. CREATE FINAL PROMPT
# --------------------------------------------------

final_prompt = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)


# --------------------------------------------------
# 12. SEND PROMPT TO GEMINI
# --------------------------------------------------

response = llm.invoke(final_prompt)


# --------------------------------------------------
# 13. PRINT ANSWER
# --------------------------------------------------

print("\nAnswer:")
print(response.text)