from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load document
loader = TextLoader("../../data/knowledge.txt")

documents = loader.load()


# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=20
)


# Split document
chunks = text_splitter.split_documents(documents)


print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)