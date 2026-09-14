from langchain_core.documents import Document

document = Document(
    page_content="Deep learning is a subfield of machine learning."
)

print(document)
print(document.page_content)