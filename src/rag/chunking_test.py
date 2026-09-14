def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def chunk_text(text, chunk_size=100):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


text = load_document("../../data/knowledge.txt")

chunks = chunk_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)