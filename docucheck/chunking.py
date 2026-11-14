"""Step 2 of the pipeline: split page text into overlapping chunks.

Every chunk follows the format the team agreed on:
    {"chunk_id": int, "doc_name": str, "page": int, "text": str}

Chunks never cross a page boundary, so each chunk has exactly one page number
to cite. Sizes are measured in characters: the default embedding model reads
at most about 256 tokens (roughly 1,000 characters), so longer chunks would be
cut off silently.
"""

from __future__ import annotations

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150


def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Split text into pieces of at most chunk_size characters.

    Consecutive pieces share about `overlap` characters, so a sentence that
    falls on a boundary still appears whole in at least one chunk. Splits
    happen at spaces where possible, so words are not cut in half.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    pieces = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Move the end back to the last space so we do not split a word.
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        # Step back by `overlap` characters, then forward to the next word start.
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return pieces


def chunk_pages(pages: list[dict], doc_name: str,
                chunk_size: int = DEFAULT_CHUNK_SIZE,
                overlap: int = DEFAULT_OVERLAP,
                start_id: int = 0) -> list[dict]:
    """Turn [{"page", "text"}, ...] from ingest.extract_pages into chunks."""
    chunks = []
    chunk_id = start_id
    for page in pages:
        for piece in split_text(page["text"], chunk_size, overlap):
            chunks.append({
                "chunk_id": chunk_id,
                "doc_name": doc_name,
                "page": page["page"],
                "text": piece,
            })
            chunk_id += 1
    return chunks
