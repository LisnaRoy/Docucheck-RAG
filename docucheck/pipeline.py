from docucheck.llm import generate
from docucheck.prompts import SYSTEM_PROMPT, build_prompt
from docucheck.retriever import retrieve

NOT_FOUND_TEXT = "not found in the document"
NOT_FOUND_ANSWER = "Not found in the document."


def get_chunks(question: str, top_k: int = 5) -> list[dict]:
    return retrieve(question, top_k=top_k)


def ask(question: str) -> dict:
    """Return {"answer": str, "sources": [chunk, ...], "found": bool}."""
    chunks = get_chunks(question)
    if not chunks:
        return {"answer": NOT_FOUND_ANSWER, "sources": [], "found": False}
    answer = generate(build_prompt(question, chunks), SYSTEM_PROMPT).strip()
    found = NOT_FOUND_TEXT not in answer.lower()
    return {"answer": answer, "sources": chunks, "found": found}