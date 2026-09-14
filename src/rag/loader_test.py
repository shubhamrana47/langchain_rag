from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../data/knowledge.txt")

documents = loader.load()

print("Number of documents:", len(documents))

print("\nDocument content:")
print(documents[0].page_content)

print("\nMetadata:")
print(documents[0].metadata)