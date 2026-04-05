import streamlit as st

from src.ocr_indexer import ocr_and_index_pdf
from rag_groq import run_rag

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Biomedical RAG Assistant",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Biomedical Research Assistant")
st.caption("OCR-based RAG with inline citations (Groq-powered)")


# -----------------------------
# Sidebar: PDF Upload
# -----------------------------
st.sidebar.header("📄 Upload PubMed PDF")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a biomedical research PDF",
    type=["pdf"],
)

if uploaded_pdf:
    with st.sidebar.spinner("Running OCR and updating knowledge base..."):
        try:
            num_chunks = ocr_and_index_pdf(uploaded_pdf)
            st.sidebar.success(f"Indexed {num_chunks} new chunks")
        except Exception as e:
            st.sidebar.error(f"Failed to index PDF: {str(e)}")


# -----------------------------
# Session state
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# User input
# -----------------------------
query = st.chat_input("Ask a biomedical research question...")

# -----------------------------
# Handle user query
# -----------------------------
if query:
    # Append user message
    st.session_state.chat_history.append(("user", query))

    with st.spinner("Searching literature and generating answer..."):
        try:
            answer_lines = run_rag(query, k=6)
        except Exception as e:
            answer_lines = [f"Failed to generate answer: {str(e)}"]

    # Force fresh assistant response
    st.session_state.chat_history.append(
        ("assistant", list(answer_lines))
    )
if st.sidebar.button("🧹 Clear chat"):
    st.session_state.chat_history = []


# -----------------------------
# Display chat history
# -----------------------------
for role, content in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)

    elif role == "assistant":
        with st.chat_message("assistant"):
            for line in content:
                st.markdown(line)

            # Expandable sources
            with st.expander("Sources"):
                for line in content:
                    if "[" in line and "]" in line:
                        st.markdown(f"- {line.split('[')[-1].strip(']')}")
