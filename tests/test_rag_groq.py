import importlib
import sys
from types import SimpleNamespace


class _StubRetriever:
    def invoke(self, question):
        return []


class _StubChroma:
    def __init__(self, *args, **kwargs):
        pass

    def as_retriever(self, search_kwargs=None):
        return _StubRetriever()


class _StubEmbeddings:
    def __init__(self, *args, **kwargs):
        pass


sys.modules.setdefault("langchain_huggingface", SimpleNamespace(HuggingFaceEmbeddings=_StubEmbeddings))
sys.modules.setdefault("langchain_chroma", SimpleNamespace(Chroma=_StubChroma))

rag_groq = importlib.import_module("rag_groq")


class _FakeEmbedder:
    def __init__(self, *args, **kwargs):
        pass

    def embed_documents(self, docs):
        return [[1.0, 0.0] for _ in docs]

    def embed_query(self, query):
        return [1.0, 0.0]


def test_split_sentences_filters_short_entries():
    text = "Short. This sentence should be kept because it is long enough. Ok."
    result = rag_groq.split_sentences(text)
    assert len(result) == 1
    assert "long enough" in result[0]


def test_attach_inline_citations_handles_missing_page(monkeypatch):
    monkeypatch.setattr(rag_groq, "HuggingFaceEmbeddings", _FakeEmbedder)

    docs = [SimpleNamespace(page_content="doc", metadata={"filename": "paper.pdf"})]
    result = rag_groq.attach_inline_citations(["Evidence statement about treatment outcomes."], docs)

    assert result[0].endswith("[paper.pdf]")


def test_run_rag_returns_no_docs_message(monkeypatch):
    fake_retriever = SimpleNamespace(invoke=lambda q: [])
    monkeypatch.setattr(rag_groq, "load_retriever", lambda k: fake_retriever)

    result = rag_groq.run_rag("question", k=3)
    assert result == ["No relevant information was found in the indexed documents."]
