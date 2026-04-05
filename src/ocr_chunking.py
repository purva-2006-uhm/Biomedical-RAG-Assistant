# src/ocr_chunking.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ocr_loader import load_all_pdfs_ocr


def chunk_ocr_documents():
    documents = load_all_pdfs_ocr()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(documents)

    print(f"Total OCR chunks created: {len(chunks)}")

    return chunks


if __name__ == "__main__":
    chunks = chunk_ocr_documents()

    print("\n--- Sample OCR Chunk ---\n")
    print(chunks[0].page_content[:500])
    print("\nMetadata:", chunks[0].metadata)
