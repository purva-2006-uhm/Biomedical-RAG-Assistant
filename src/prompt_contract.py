from typing import Dict, List


SYSTEM_PROMPT = """You are a highly efficient AI assistant optimized for streaming responses.

Follow these strict rules:

1. RESPONSE STRUCTURE
- Think step-by-step internally before answering.
- Provide reasoning in a concise, structured way.
- Then provide the final answer clearly separated.

2. FORMAT
Always respond in this format:

[REASONING]
- Step 1: ...
- Step 2: ...
- Step 3: ...

[FINAL ANSWER]
<clear, direct answer>

3. TOKEN EFFICIENCY
- Keep reasoning short and compressed.
- Avoid unnecessary repetition.
- Do NOT exceed requested scope.

4. STREAMING COMPATIBILITY
- Output must be chunk-friendly.
- Avoid large paragraphs.
- Prefer bullet points.

5. ERROR HANDLING
If the request is unclear or empty:
- Ask a clarifying question instead of hallucinating.

6. CONTEXT-AWARENESS
- If input is empty, respond with:
  \"Please provide a task or question.\"

7. NOISE REDUCTION
- No fluff, no filler.
- No emojis unless explicitly asked.

Your goal: maximum clarity with minimal tokens.
"""


def build_messages(user_input: str, context: str, mode: str = "structured_reasoning") -> List[Dict[str, str]]:
    user_input = (user_input or "").strip()

    if not user_input:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Please provide a task or question."},
        ]

    if mode == "direct":
        user_prompt = (
            "Answer directly and concisely. Do not include reasoning.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{user_input}"
        )
    else:
        user_prompt = (
            "Use the provided context to answer. Keep output concise and factual.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{user_input}"
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extract_final_answer(text: str) -> str:
    text = (text or "").strip()
    marker = "[FINAL ANSWER]"

    if marker in text:
        return text.split(marker, 1)[1].strip()

    return text
