"""Step 1 of the pipeline: read a PDF and return its text page by page."""

from __future__ import annotations

import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Tidy raw PDF text so chunks are easier to read and embed."""
    # Join words split across lines with a hyphen: "regis-\ntration" -> "registration".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Turn single line breaks into spaces but keep blank lines as paragraph breaks.
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    # Collapse repeated spaces and tabs.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """Return [{"page": int, "text": str}, ...] with 1-based page numbers.

    Page numbers match what a PDF viewer shows as the page index, so citations
    can be checked by opening the PDF. Pages without text (for example scanned
    images) are kept with empty text so the numbering never shifts.
    """
    import pymupdf  # imported here so the rest of the package works without it

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    with pymupdf.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            pages.append({"page": index + 1, "text": clean_text(page.get_text("text"))})
    return pages
