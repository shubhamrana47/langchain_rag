import chromadb
from pathlib import Path


# ============================================================
# CHROMA PATH
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

PERSIST_DIR = (
    PROJECT_ROOT
    / "storage"
    / "chroma_history_huggingface"
)

COLLECTION_NAME = "langchain"


# ============================================================
# CONNECT TO CHROMA
# ============================================================

client = chromadb.PersistentClient(
    path=str(PERSIST_DIR)
)

collection = client.get_collection(
    COLLECTION_NAME
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("=" * 70)
print("CHROMA DATABASE INSPECTION")
print("=" * 70)

print(f"\nDatabase path:")
print(PERSIST_DIR)

print(f"\nCollection:")
print(COLLECTION_NAME)

print(f"\nTotal records:")
print(collection.count())


# ============================================================
# GET RECORDS
# ============================================================

data = collection.get(
    include=[
        "documents",
        "embeddings",
        "metadatas"
    ]
)


# ============================================================
# PRINT FIRST RECORD
# ============================================================

print("\n" + "=" * 70)
print("FIRST RECORD")
print("=" * 70)

record_id = data["ids"][0]

document = data["documents"][0]

embedding = data["embeddings"][0]

metadata = data["metadatas"][0]


print("\nID:")
print(record_id)


print("\nDOCUMENT / CHUNK:")
print(document)


print("\nMETADATA:")
print(metadata)


print("\nEMBEDDING:")
print(embedding)


print("\nEMBEDDING LENGTH:")
print(len(embedding))


# ============================================================
# SHOW ALL RECORD IDs
# ============================================================

print("\n" + "=" * 70)
print("ALL RECORD IDS")
print("=" * 70)

for i, record_id in enumerate(data["ids"]):

    print(
        f"{i + 1}. {record_id}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Total records: {len(data['ids'])}"
)

print(
    f"First document length: {len(document)} characters"
)

print(
    f"Embedding dimensions: {len(embedding)}"
)