from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# --------------------------------------------------
# 1. Project path
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]


# --------------------------------------------------
# 2. Embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 3. Connect to Chroma
# --------------------------------------------------

vectorstore = Chroma(
    persist_directory=str(
        BASE_DIR / "storage" / "chroma_history_huggingface"
    ),
    collection_name="langchain",
    embedding_function=embeddings
)


# --------------------------------------------------
# 4. Chroma Debug
# --------------------------------------------------

print("\n--- CHROMA DEBUG ---")

print(
    "Database path:",
    BASE_DIR / "storage" / "chroma_history_huggingface"
)

print(
    "Collection:",
    vectorstore._collection.name
)

print(
    "Total records:",
    vectorstore._collection.count()
)


# --------------------------------------------------
# 5. Create retriever
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# 6. Gemini LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# --------------------------------------------------
# 7. Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant answering questions about a PDF document.

Use the provided context to answer the user's question.

If the answer is not available in the context, say:
"I don't have enough information in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
)


# --------------------------------------------------
# 8. RAG function
# --------------------------------------------------

def rag_chain(question):

    # Retrieve relevant PDF chunks
    documents = retriever.invoke(question)

    print("\n--- RETRIEVED DOCUMENTS ---")

    for i, document in enumerate(documents):

        print(f"\nDocument {i + 1}:")

        print("CONTENT:")
        print(document.page_content)

        print("METADATA:")
        print(document.metadata)


    # Combine chunks into context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    print("\n--- CONTEXT ---")
    print(context)


    # Create final prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })


    # Send prompt to Gemini
    response = llm.invoke(final_prompt)


    return response.text


# --------------------------------------------------
# 9. Run
# --------------------------------------------------

if __name__ == "__main__":

    question = input("Ask a question: ")

    answer = rag_chain(question)

    print("\nAnswer:")
    print(answer)