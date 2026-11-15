SYSTEM_PROMPT = """You answer questions using ONLY the passages provided.

Rules:
- Use only the information in the passages. Do not use outside knowledge.
- Cite the page number for every fact, in the form (page 12). Put the citation at the end of the sentence or list item.
- If the passages do not contain the answer, reply with exactly this sentence and nothing else:
  Not found in the document.
- Write lists as separate lines, each starting with "- ".
- Keep the answer short and factual."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    passages = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in chunks)
    return f"Passages:\n{passages}\n\nQuestion: {question}\n\nAnswer:"