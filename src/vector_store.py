# src/vector_store_ocr.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ocr_chunking import chunk_ocr_documents

CHROMA_DIR = "chroma_db"


def sanitize_metadata(metadata: dict) -> dict:
    """
    Keep only ChromaDB-safe metadata
    """
    allowed_keys = {"filename", "page_number", "source", "loader"}
    return {
        k: metadata[k]
        for k in metadata
        if k in allowed_keys and isinstance(metadata[k], (str, int))
    }


def build_ocr_vector_store():
    print("Building ChromaDB using OCR chunks...")

    chunks = chunk_ocr_documents()
    print(f"Total OCR chunks: {len(chunks)}")

    clean_chunks = []
    for chunk in chunks:
        clean_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=sanitize_metadata(chunk.metadata),
            )
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    vectordb = Chroma.from_documents(
        documents=clean_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print("✅ OCR-based ChromaDB indexing complete")
    return vectordb


if __name__ == "__main__":
    build_ocr_vector_store()
