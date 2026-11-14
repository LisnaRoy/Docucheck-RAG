"""Step 4 of the pipeline: store chunk vectors in FAISS and search them.

IndexFlatIP compares the query with every stored vector using the inner
(dot) product. Because the vectors are normalised, that is exact cosine
similarity. A flat index is brute force, which is fast enough for documents
with up to tens of thousands of chunks.

On disk an index is a folder with two files:
    index.faiss  the vectors
    chunks.json  the chunks (row i of the index is chunks[i]) plus metadata
"""

from __future__ import annotations

import json
from pathlib import Path

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


def build_index(embeddings):
    """Create an inner-product index from an (n, dim) float32 array."""
    import faiss
    import numpy as np

    embeddings = np.ascontiguousarray(embeddings, dtype="float32")
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(index, query_vector, top_k: int = 5) -> list[tuple[int, float]]:
    """Return [(row, score), ...] for the top_k most similar rows, best first."""
    import numpy as np

    query = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
    top_k = min(top_k, index.ntotal)
    if top_k <= 0:
        return []
    scores, rows = index.search(query, top_k)
    # FAISS pads with -1 when there are fewer results than asked for.
    return [(int(r), float(s)) for r, s in zip(rows[0], scores[0]) if r != -1]


def save_index(index, chunks: list[dict], index_dir: str | Path,
               metadata: dict | None = None) -> Path:
    """Write the index and its chunks to index_dir (created if missing)."""
    import faiss

    if index.ntotal != len(chunks):
        raise ValueError(f"index has {index.ntotal} vectors but there are {len(chunks)} chunks")
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    # faiss.write_index can fail on Windows paths with non-ASCII characters,
    # so we serialise to bytes and write the file with Python instead.
    (index_dir / INDEX_FILE).write_bytes(faiss.serialize_index(index).tobytes())
    payload = {"metadata": metadata or {}, "chunks": chunks}
    (index_dir / CHUNKS_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return index_dir


def load_index(index_dir: str | Path):
    """Return (index, chunks, metadata) saved by save_index."""
    import faiss
    import numpy as np

    index_dir = Path(index_dir)
    index_path, chunks_path = index_dir / INDEX_FILE, index_dir / CHUNKS_FILE
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"No index in {index_dir}. Build one first, for example:\n"
            f"  python -m docucheck.retriever build data/pdfs/<file>.pdf"
        )
    index = faiss.deserialize_index(np.frombuffer(index_path.read_bytes(), dtype="uint8"))
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    return index, payload["chunks"], payload.get("metadata", {})
