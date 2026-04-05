from src.prompt_contract import build_messages, extract_final_answer


def test_build_messages_returns_prompt_for_empty_user_input():
    messages = build_messages("", "ctx")
    assert len(messages) == 2
    assert messages[1]["content"] == "Please provide a task or question."


def test_build_messages_direct_mode_instruction_present():
    messages = build_messages("What is RAG?", "context text", mode="direct")
    assert messages[0]["role"] == "system"
    assert "Do not include reasoning" in messages[1]["content"]


def test_extract_final_answer_marker_split():
    text = "[REASONING]\n- Step 1\n\n[FINAL ANSWER]\nRAG is retrieval-augmented generation."
    assert extract_final_answer(text) == "RAG is retrieval-augmented generation."
