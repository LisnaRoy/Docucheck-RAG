SYSTEM_PROMPT = """You answer questions using ONLY the passages provided.
Rules:
- Use only the information in the passages. Do not use outside knowledge.
- Cite the page number for every claim, like (page 12).
- If the passages do not contain the answer, reply exactly:
  "Not found in the document."
- Keep the answer short and factual."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    passages = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in chunks)
    return f"Passages:\n{passages}\n\nQuestion: {question}\n\nAnswer:"