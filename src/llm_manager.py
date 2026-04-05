import time
import importlib
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from src.config import (
    GROQ_API_KEY,
    LLM_BASE_URL,
    LLM_CONTEXT_WINDOW,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    NVIDIA_API_KEY,
)

@dataclass
class LLMResult:
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[int] = None


def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    joined = "\n".join(m.get("content", "") for m in messages)
    return max(1, len(joined) // 4)


class LLMManager:
    _client = None

    @classmethod
    def _get_client(cls):
        if cls._client is not None:
            return cls._client

        if LLM_PROVIDER == "groq":
            groq_module = importlib.import_module("groq")
            Groq = getattr(groq_module, "Groq")

            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq")

            cls._client = Groq(api_key=GROQ_API_KEY)
            return cls._client

        if LLM_PROVIDER == "nvidia":
            openai_module = importlib.import_module("openai")
            OpenAI = getattr(openai_module, "OpenAI")

            if not NVIDIA_API_KEY:
                raise RuntimeError("NVIDIA_API_KEY is required when LLM_PROVIDER=nvidia")

            cls._client = OpenAI(api_key=NVIDIA_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT_SECONDS)
            return cls._client

        raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

    @staticmethod
    def _resolve_output_budget(messages: List[Dict[str, str]], requested_max_tokens: Optional[int]) -> int:
        max_out = requested_max_tokens or LLM_MAX_OUTPUT_TOKENS
        input_est = estimate_tokens(messages)
        available = LLM_CONTEXT_WINDOW - input_est

        if available <= 0:
            raise ValueError("Input is too large for the configured context window.")

        return max(128, min(max_out, available))

    @staticmethod
    def _extract_usage(completion) -> Dict[str, Optional[int]]:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    @classmethod
    def complete(
        cls,
        messages: List[Dict[str, str]],
        requested_max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResult:
        client = cls._get_client()
        max_tokens = cls._resolve_output_budget(messages, requested_max_tokens)
        temp = LLM_TEMPERATURE if temperature is None else temperature

        last_error = None

        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                started = time.time()

                kwargs = {
                    "model": LLM_MODEL,
                    "messages": messages,
                    "temperature": temp,
                }

                if LLM_PROVIDER == "groq":
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens

                completion = client.chat.completions.create(**kwargs)
                latency_ms = int((time.time() - started) * 1000)

                text = completion.choices[0].message.content.strip()
                usage = cls._extract_usage(completion)

                return LLMResult(
                    text=text,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    latency_ms=latency_ms,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= LLM_MAX_RETRIES:
                    break
                time.sleep(min(2 ** attempt, 4))

        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    @classmethod
    def stream(cls, messages: List[Dict[str, str]], requested_max_tokens: Optional[int] = None) -> Iterator[str]:
        client = cls._get_client()
        max_tokens = cls._resolve_output_budget(messages, requested_max_tokens)

        kwargs = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "stream": True,
        }

        if LLM_PROVIDER == "groq":
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        stream = client.chat.completions.create(**kwargs)

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield reasoning

            content = getattr(delta, "content", None)
            if content:
                yield content
