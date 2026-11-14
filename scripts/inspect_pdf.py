"""Check how a PDF is extracted and chunked before indexing it.

Usage:
    python scripts/inspect_pdf.py data/pdfs/handbook.pdf
    python scripts/inspect_pdf.py data/pdfs/handbook.pdf --chunk-size 300 --show 5

Compare the printed page numbers and text with the PDF viewer. Look for pages
with no text (scanned images), broken words, or headers repeated on every page.
"""

import argparse
import sys
from pathlib import Path

# Let the script import the docucheck package when run from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docucheck.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_pages  # noqa: E402
from docucheck.ingest import extract_pages  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--show", type=int, default=3, help="number of sample chunks to print")
    args = parser.parse_args()

    pages = extract_pages(args.pdf)
    chunks = chunk_pages(pages, Path(args.pdf).name, args.chunk_size, args.overlap)

    empty = [p["page"] for p in pages if not p["text"]]
    print(f"File:   {args.pdf}")
    print(f"Pages:  {len(pages)}  (without text: {empty or 'none'})")
    print(f"Chunks: {len(chunks)}  (size {args.chunk_size}, overlap {args.overlap})")
    if chunks:
        lengths = [len(c["text"]) for c in chunks]
        print(f"Chunk length: min {min(lengths)}, avg {sum(lengths) // len(lengths)}, "
              f"max {max(lengths)} characters")

    print("\nCharacters per page:")
    for p in pages:
        print(f"  page {p['page']:>3}: {len(p['text']):>6}")

    for c in chunks[: args.show]:
        print(f"\n--- chunk {c['chunk_id']} (page {c['page']}) ---\n{c['text']}")


if __name__ == "__main__":
    main()
