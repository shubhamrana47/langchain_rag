from pathlib import Path


def detect_file_type(file_path):
    extension = Path(file_path).suffix.lower()

    if extension == ".txt":
        return "text"

    elif extension == ".pdf":
        return "pdf"

    elif extension in [".jpg", ".jpeg"]:
        return "image"

    else:
        return "unsupported"


def process_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


if __name__ == "__main__":
    file_path = input("Enter file path: ")

    file_type = detect_file_type(file_path)

    print(f"Detected file type: {file_type}")

    if file_type == "text":
        text = process_text_file(file_path)

        print("\nExtracted text:")
        print(text)

    elif file_type == "pdf":
        print("PDF processor will be added next.")

    elif file_type == "image":
        print("Image processor will be added next.")

    else:
        print("Unsupported file type.")