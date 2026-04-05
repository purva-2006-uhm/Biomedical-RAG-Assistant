
# src/retrieval.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL


def load_retriever(k: int = 3):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(20, k * 3)}
    )

    return retriever


def similarity_search(query: str, k: int = 3):
    retriever = load_retriever(k=k)
    return retriever.invoke(query)



