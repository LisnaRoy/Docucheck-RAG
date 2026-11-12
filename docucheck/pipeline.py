from docucheck.llm import generate
from docucheck.prompts import SYSTEM_PROMPT, build_prompt

NOT_FOUND_TEXT = "not found in the document"

# Placeholder until Person A's retriever is ready (week 3).
FAKE_CHUNKS = [
    {"chunk_id": 0, "doc_name": "test.pdf", "page": 3,
     "text": "Students must register their address within 14 days of moving."},
    {"chunk_id": 1, "doc_name": "test.pdf", "page": 5,
     "text": "The library is open Monday to Friday from 8:00 to 18:00."},
    {"chunk_id": 2, "doc_name": "test.pdf", "page": 8,
     "text": "Exam registration closes four weeks before the exam date."},
]


def get_chunks(question: str, top_k: int = 5) -> list[dict]:
    # Week 3: replace this body with
    #   from docucheck.retriever import retrieve
    #   return retrieve(question, top_k=top_k)
    return FAKE_CHUNKS


def ask(question: str) -> dict:
    """Return {"answer": str, "sources": [chunk, ...], "found": bool}."""
    chunks = get_chunks(question)
    answer = generate(build_prompt(question, chunks), SYSTEM_PROMPT).strip()
    found = NOT_FOUND_TEXT not in answer.lower()
    return {"answer": answer, "sources": chunks, "found": found}