# src/rag_groq.py

import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, LLM_MAX_OUTPUT_TOKENS, LLM_RESPONSE_MODE
from src.llm_manager import LLMManager
from src.prompt_contract import build_messages, extract_final_answer

# -----------------------------
# 1. Load retriever
# -----------------------------
def load_retriever(k: int = 5):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    return vectordb.as_retriever(search_kwargs={"k": k})


# -----------------------------
# 2. Ask Groq (sentence-only)
# -----------------------------
def ask_llm_sentences(question: str, context: str):
    messages = build_messages(question, context, mode=LLM_RESPONSE_MODE)
    result = LLMManager.complete(messages, requested_max_tokens=LLM_MAX_OUTPUT_TOKENS)
    return extract_final_answer(result.text)


# -----------------------------
# 3. Sentence splitting
# -----------------------------
def split_sentences(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# -----------------------------
# 4. Inline citations
# -----------------------------
def attach_inline_citations(sentences, retrieved_docs):
    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    doc_texts = [doc.page_content for doc in retrieved_docs]
    doc_embeddings = embedder.embed_documents(doc_texts)

    cited = []

    for sentence in sentences:
        sent_emb = embedder.embed_query(sentence)

        scores = [
            sum(a * b for a, b in zip(sent_emb, doc_emb))
            for doc_emb in doc_embeddings
        ]

        if not scores:
            cited.append(f"{sentence} [No source found]")
            continue

        best_idx = scores.index(max(scores))
        best_doc = retrieved_docs[best_idx]

        filename = best_doc.metadata.get("filename", "unknown")
        page_number = best_doc.metadata.get("page_number")

        if page_number is None:
            citation = f"[{filename}]"
        else:
            citation = f"[{filename}, p.{page_number}]"

        cited.append(f"{sentence} {citation}")

    return cited


# -----------------------------
# 5. Full RAG
# -----------------------------
def run_rag(question: str, k: int = 5):
    retriever = load_retriever(k)
    docs = retriever.invoke(question)

    if not docs:
        return ["No relevant information was found in the indexed documents."]

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    raw = ask_llm_sentences(question, context)
    sentences = split_sentences(raw)

    return attach_inline_citations(sentences, docs)


# -----------------------------
# 6. Test
# -----------------------------
if __name__ == "__main__":
    q = "What is the effectiveness of ribavirin for treating Lassa fever?"
    for line in run_rag(q, k=6):
        print("-", line)
