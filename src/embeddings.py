from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.ocr_chunking import chunk_ocr_documents

from src.ocr_chunking import chunk_ocr_documents



def generate_embeddings(chunks: List):
    """
    Generate normalized embeddings for document chunks
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    texts = [chunk.page_content for chunk in chunks]
    vectors = embedding_model.embed_documents(texts)

    return vectors


if __name__ == "__main__":
    documents = load_all_pdfs()
    chunks = chunk_documents(documents)

    print(f"Total chunks to embed: {len(chunks)}")

    vectors = generate_embeddings(chunks)

    print(f"Total vectors generated: {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")

    print("\n--- Sample Vector (first 10 values) ---")
    print(vectors[0][:10])
