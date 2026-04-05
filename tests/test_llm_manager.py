from types import SimpleNamespace

import src.llm_manager as llm_manager


def _completion(text="ok", prompt_tokens=10, completion_tokens=12, total_tokens=22):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


class _FakeClient:
    def __init__(self, completion):
        self._completion = completion
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return self._completion


def test_estimate_tokens_has_minimum_of_one():
    assert llm_manager.estimate_tokens([{"content": ""}]) == 1


def test_resolve_output_budget_clamps_to_context_window(monkeypatch):
    monkeypatch.setattr(llm_manager, "LLM_CONTEXT_WINDOW", 200)
    messages = [{"content": "a" * 600}]  # ~150 tokens

    budget = llm_manager.LLMManager._resolve_output_budget(messages, requested_max_tokens=300)
    assert budget == 128


def test_complete_returns_usage_and_text(monkeypatch):
    monkeypatch.setattr(llm_manager, "LLM_PROVIDER", "nvidia")
    monkeypatch.setattr(llm_manager, "LLM_MODEL", "dummy-model")

    fake = _FakeClient(_completion(text="hello"))
    monkeypatch.setattr(llm_manager.LLMManager, "_get_client", classmethod(lambda cls: fake))

    result = llm_manager.LLMManager.complete([{"role": "user", "content": "hi"}], requested_max_tokens=256)

    assert result.text == "hello"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 12
    assert result.total_tokens == 22


def test_stream_yields_reasoning_and_content(monkeypatch):
    monkeypatch.setattr(llm_manager, "LLM_PROVIDER", "nvidia")

    chunk1 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content="R1 ", content=None))])
    chunk2 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content=None, content="A1"))])

    def create_stream(**kwargs):
        return [chunk1, chunk2]

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_stream)))
    monkeypatch.setattr(llm_manager.LLMManager, "_get_client", classmethod(lambda cls: fake))

    output = "".join(llm_manager.LLMManager.stream([{"role": "user", "content": "x"}], requested_max_tokens=128))
    assert output == "R1 A1"
