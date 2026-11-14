"""Step 5 of the pipeline: find the chunks that best match a question.

Two entry points:
    index_pdf(pdf_path)          PDF -> pages -> chunks -> embeddings -> saved index
    retrieve(question, top_k=5)  question -> embedding -> top_k chunks from the index

retrieve() returns chunks in the agreed format plus a "score" field (cosine
similarity, higher is more similar), best match first. Person B's pipeline
calls it to get the passages for the LLM.

Command line, for testing from the terminal:
    python -m docucheck.retriever build data/pdfs/handbook.pdf
    python -m docucheck.retriever ask "When does exam registration close?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docucheck import embeddings, vector_store
from docucheck.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_pages
from docucheck.ingest import extract_pages

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"

# Loaded indexes, keyed by folder, so repeated questions do not reload from disk.
_cache: dict[str, tuple] = {}


def build_from_chunks(chunks: list[dict], index_dir: str | Path = DEFAULT_INDEX_DIR,
                      model_name: str = embeddings.DEFAULT_MODEL,
                      extra_metadata: dict | None = None) -> int:
    """Embed chunks, build the FAISS index and save it. Returns the chunk count."""
    if not chunks:
        raise ValueError("No text found to index. Is the PDF scanned (image only)?")
    vectors = embeddings.embed_texts([c["text"] for c in chunks], model_name)
    index = vector_store.build_index(vectors)
    metadata = {"model_name": model_name, **(extra_metadata or {})}
    vector_store.save_index(index, chunks, index_dir, metadata)
    _cache.pop(str(Path(index_dir).resolve()), None)  # drop any stale copy
    return len(chunks)


def index_pdf(pdf_path: str | Path, index_dir: str | Path = DEFAULT_INDEX_DIR,
              chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP,
              model_name: str = embeddings.DEFAULT_MODEL) -> int:
    """Read a PDF and build a fresh index for it. Returns the chunk count."""
    pdf_path = Path(pdf_path)
    chunks = chunk_pages(extract_pages(pdf_path), pdf_path.name, chunk_size, overlap)
    return build_from_chunks(chunks, index_dir, model_name, {
        "doc_name": pdf_path.name, "chunk_size": chunk_size, "overlap": overlap,
    })


def _load(index_dir: str | Path) -> tuple:
    key = str(Path(index_dir).resolve())
    if key not in _cache:
        _cache[key] = vector_store.load_index(index_dir)
    return _cache[key]


def retrieve(question: str, top_k: int = 5,
             index_dir: str | Path = DEFAULT_INDEX_DIR) -> list[dict]:
    """Return up to top_k chunks most similar to the question, best first."""
    if not question.strip():
        return []
    index, chunks, metadata = _load(index_dir)
    # Embed the question with the same model that embedded the chunks;
    # vectors from different models cannot be compared.
    model_name = metadata.get("model_name", embeddings.DEFAULT_MODEL)
    query_vector = embeddings.embed_texts([question], model_name, is_query=True)[0]
    return [{**chunks[row], "score": score}
            for row, score in vector_store.search(index, query_vector, top_k)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build or query the DocuCheck index.")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="index a PDF")
    build.add_argument("pdf")
    build.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    build.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    build.add_argument("--model", default=embeddings.DEFAULT_MODEL)

    ask = sub.add_parser("ask", help="show the chunks retrieved for a question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args(argv)
    if args.command == "build":
        count = index_pdf(args.pdf, args.index_dir, args.chunk_size, args.overlap, args.model)
        print(f"Indexed {count} chunks from {args.pdf} into {args.index_dir}")
    else:
        for rank, chunk in enumerate(retrieve(args.question, args.top_k, args.index_dir), 1):
            preview = chunk["text"][:200].replace("\n", " ")
            print(f"{rank}. page {chunk['page']}  score {chunk['score']:.3f}\n   {preview}\n")


if __name__ == "__main__":
    main()
