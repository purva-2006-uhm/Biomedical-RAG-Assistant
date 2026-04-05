import tempfile
from pathlib import Path
import os

from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import COLLECTION_NAME, EMBED_MODEL


CHROMA_DIR = "chroma_db"



def ocr_and_index_pdf(uploaded_file):
    """
    OCR → chunk → embed → add to ChromaDB
    """

    # -----------------------------
    # 1. Save uploaded PDF temporarily
    # -----------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # -----------------------------
    # 2. OCR extraction
    # -----------------------------
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="ocr_only",
            infer_table_structure=False,
        )
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    docs = []
    for el in elements:
        if el.text and len(el.text.strip()) > 50:
            docs.append(
                Document(
                    page_content=el.text,
                    metadata={
                        "filename": uploaded_file.name,
                        "category": el.category,
                    },
                )
            )

    # -----------------------------
    # 3. Chunking
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(docs)

    # -----------------------------
    # 4. Embedding + Chroma append
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )



    vectordb.add_documents(chunks)

    return len(chunks)
