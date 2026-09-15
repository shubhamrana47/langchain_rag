
import os
import mimetypes
from pathlib import Path

import pymupdf
import chromadb

from dotenv import load_dotenv
from google import genai
from google.genai import types

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# File to ingest
FILE_PATH = PROJECT_ROOT / "data" / "pdfsamplefortesting.pdf"

# Chroma persistent storage
PERSIST_DIR = PROJECT_ROOT / "storage" / "chroma_history_huggingface"

# Chroma collection name
COLLECTION_NAME = "langchain"


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# FILE TYPE DETECTION
# ============================================================

def detect_file_type(file_path):
    """
    Detect the type of input file.

    Supported:
        .txt
        .pdf
        .jpg
        .jpeg
    """

    extension = Path(file_path).suffix.lower()

    if extension == ".txt":
        return "text"

    elif extension == ".pdf":
        return "pdf"

    elif extension in [".jpg", ".jpeg"]:
        return "image"

    else:
        return "unsupported"


# ============================================================
# TEXT FILE PROCESSING
# ============================================================

def process_text_file(file_path):
    """
    Read a normal TXT file and convert it
    into a LangChain Document.
    """

    print("\nProcessing TXT file...")

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    if not text.strip():
        print("Warning: TXT file is empty.")

        return []

    document = Document(
        page_content=text,
        metadata={
            "source": str(file_path),
            "type": "text"
        }
    )

    print("Text file processed successfully.")

    return [document]


# ============================================================
# GEMINI VISION PROCESSING
# ============================================================

def process_image_with_vision(
    image_bytes,
    mime_type,
    source,
    page=None
):
    """
    Send an image to Gemini Vision and convert
    the image understanding into a LangChain Document.
    """

    print("    Sending image to Gemini Vision...")

    prompt = """
Analyze this image carefully.

Extract all useful information from the image that could
help answer questions about it later in a RAG system.

If the image contains:

- text → extract the important text
- tables → describe the table contents
- charts or graphs → explain the important information
- maps → explain what the map shows
- diagrams → explain the relationships shown
- photographs → describe relevant information

Return a clear text-based description suitable for
storing as knowledge in a RAG system.

Do not invent information that is not visible in the image.
Only describe information that can reasonably be determined
from the image.
"""

    # Convert raw image bytes into a Gemini image Part
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )

    # Send image + prompt to Gemini Vision
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=[
            image_part,
            prompt
        ]
    )

    image_text = response.text

    if not image_text:
        image_text = (
            "No useful information could be extracted "
            "from this image."
        )

    metadata = {
        "source": str(source),
        "type": "image"
    }

    if page is not None:
        metadata["page"] = page

    return Document(
        page_content=image_text,
        metadata=metadata
    )


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf_file(file_path):
    """
    Process a PDF.

    For every page:

    1. Extract normal text.
    2. Extract embedded images.
    3. Send images to Gemini Vision.
    4. Convert everything into LangChain Documents.
    """

    print("\nProcessing PDF file...")

    pdf = pymupdf.open(file_path)

    documents = []

    print(f"Number of pages: {len(pdf)}")

    # --------------------------------------------------------
    # Process every page
    # --------------------------------------------------------

    for page_index, page in enumerate(pdf):

        page_number = page_index + 1

        print(
            f"\nProcessing page "
            f"{page_number}/{len(pdf)}..."
        )

        # ====================================================
        # EXTRACT NORMAL TEXT
        # ====================================================

        text = page.get_text()

        if text.strip():

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "page": page_number,
                        "type": "text"
                    }
                )
            )

            print("  Text found and added.")

        else:

            print("  No text found.")

        # ====================================================
        # EXTRACT IMAGES
        # ====================================================

        images = page.get_images(full=True)

        print(
            f"  Images found: {len(images)}"
        )

        # ----------------------------------------------------
        # Process every image on the page
        # ----------------------------------------------------

        for image_index, image_info in enumerate(images):

            print(
                f"  Processing image "
                f"{image_index + 1}/{len(images)}..."
            )

            # XREF identifies the image inside the PDF
            xref = image_info[0]

            # Extract image data
            image_data = pdf.extract_image(xref)

            image_bytes = image_data["image"]

            image_extension = image_data["ext"]

            # Detect MIME type
            mime_type = mimetypes.guess_type(
                f"image.{image_extension}"
            )[0]

            if mime_type is None:

                mime_type = "image/jpeg"

            # Send image to Gemini Vision
            image_document = process_image_with_vision(
                image_bytes=image_bytes,
                mime_type=mime_type,
                source=file_path,
                page=page_number
            )

            # Store Gemini's image description
            documents.append(image_document)

    # Close PDF
    pdf.close()

    return documents


# ============================================================
# JPG / JPEG PROCESSING
# ============================================================

def process_image_file(file_path):
    """
    Process a standalone JPG/JPEG image
    using Gemini Vision.
    """

    print("\nProcessing JPG/JPEG file...")

    # Detect MIME type
    mime_type = mimetypes.guess_type(
        file_path
    )[0]

    if mime_type is None:

        mime_type = "image/jpeg"

    # Read image bytes
    with open(
        file_path,
        "rb"
    ) as image_file:

        image_bytes = image_file.read()

    # Send image to Gemini Vision
    document = process_image_with_vision(
        image_bytes=image_bytes,
        mime_type=mime_type,
        source=file_path
    )

    return [document]


# ============================================================
# MAIN INGESTION
# ============================================================

def ingest():

    print("=" * 60)
    print("MULTIMODAL RAG INGESTION")
    print("=" * 60)

    # ========================================================
    # CHECK INPUT FILE
    # ========================================================

    if not FILE_PATH.exists():

        raise FileNotFoundError(
            f"\nFile not found:\n{FILE_PATH}"
        )

    print(
        f"\nInput file:\n{FILE_PATH}"
    )

    # ========================================================
    # DETECT FILE TYPE
    # ========================================================

    file_type = detect_file_type(
        FILE_PATH
    )

    print(
        f"\nDetected type: {file_type}"
    )

    if file_type == "unsupported":

        raise ValueError(
            "\nUnsupported file type.\n"
            "Supported files are:\n"
            ".txt\n"
            ".pdf\n"
            ".jpg\n"
            ".jpeg"
        )

    # ========================================================
    # PROCESS FILE
    # ========================================================

    if file_type == "text":

        documents = process_text_file(
            FILE_PATH
        )

    elif file_type == "pdf":

        documents = process_pdf_file(
            FILE_PATH
        )

    elif file_type == "image":

        documents = process_image_file(
            FILE_PATH
        )

    else:

        raise ValueError(
            "Unsupported file type."
        )

    # ========================================================
    # DOCUMENT COUNT
    # ========================================================

    print(
        f"\nTotal Documents created: "
        f"{len(documents)}"
    )

    if not documents:

        raise ValueError(
            "No content was extracted from the file."
        )

    # ========================================================
    # TEXT CHUNKING
    # ========================================================

    print(
        "\nSplitting documents into chunks..."
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Number of chunks created: "
        f"{len(chunks)}"
    )

    if not chunks:

        raise ValueError(
            "No chunks were created."
        )

    # ========================================================
    # HUGGING FACE EMBEDDINGS
    # ========================================================

    print(
        "\nInitializing HuggingFace embedding model..."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(
        "HuggingFace embedding model ready."
    )

    # ========================================================
    # CHROMA CLIENT
    # ========================================================

    print(
        "\nPreparing Chroma database..."
    )

    # Make sure storage directory exists
    PERSIST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # IMPORTANT:
    # Create the PersistentClient ourselves.
    #
    # This same client will be passed to LangChain's
    # Chroma wrapper below.
    #
    # This prevents the ingestion and query code
    # from accidentally using different Chroma instances.
    chroma_client = chromadb.PersistentClient(
        path=str(PERSIST_DIR)
    )

    # ========================================================
    # DELETE OLD COLLECTION
    # ========================================================

    try:

        chroma_client.delete_collection(
            COLLECTION_NAME
        )

        print(
            f"Cleared previous collection: "
            f"{COLLECTION_NAME}"
        )

    except Exception:

        print(
            "No previous collection found. "
            "Creating a new one."
        )

    # ========================================================
    # CREATE LANGCHAIN CHROMA VECTORSTORE
    # ========================================================

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    print(
        f"Chroma collection ready: "
        f"{COLLECTION_NAME}"
    )

    # ========================================================
    # STORE DOCUMENTS
    # ========================================================

    print(
        "\nStoring chunks and embeddings in Chroma..."
    )

    vectorstore.add_documents(
        chunks
    )

    # ========================================================
    # VERIFY DATABASE
    # ========================================================

    stored_count = (
        vectorstore._collection.count()
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "INGESTION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Documents created: {len(documents)}"
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    print(
        f"Chunks stored in Chroma: {stored_count}"
    )

    print(
        f"Chroma path: {PERSIST_DIR}"
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        "=" * 60
    )


# ============================================================
# RUN INGESTION
# ============================================================

if __name__ == "__main__":

    ingest()
