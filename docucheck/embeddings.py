"""Step 3 of the pipeline: turn text into vectors (embeddings).

Texts with similar meaning get vectors that point in similar directions. We
normalise every vector to length 1, so the dot product of two vectors equals
their cosine similarity. That is what lets the FAISS inner-product index in
vector_store.py rank chunks by cosine similarity.
"""

from __future__ import annotations

from functools import lru_cache

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # English, small and fast
MULTILINGUAL_MODEL = "intfloat/multilingual-e5-small"     # use for German PDFs


@lru_cache(maxsize=2)
def get_model(model_name: str = DEFAULT_MODEL):
    """Load a model once and reuse it; loading takes seconds, encoding is fast."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _add_prefix(texts: list[str], model_name: str, is_query: bool) -> list[str]:
    # E5 models were trained with "query: " / "passage: " prefixes and work
    # noticeably worse without them. Other models need no prefix.
    if "e5" not in model_name.lower():
        return texts
    prefix = "query: " if is_query else "passage: "
    return [prefix + t for t in texts]


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL,
                is_query: bool = False, batch_size: int = 32):
    """Return a float32 numpy array of shape (len(texts), dim), rows of length 1.

    Use is_query=True for the user's question and False for document chunks.
    """
    import numpy as np

    model = get_model(model_name)
    vectors = model.encode(
        _add_prefix(list(texts), model_name, is_query),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 100,
    )
    return np.asarray(vectors, dtype="float32")
