"""Retriever tests.

They replace the real embedding model with a tiny word-counting embedder, so
they run in a second and need no model download. They still use real FAISS,
so they are skipped when numpy or faiss-cpu is not installed.
"""

import re
import zlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faiss")

from docucheck import embeddings, retriever, vector_store  # noqa: E402

DIM = 256

CHUNKS = [
    {"chunk_id": 0, "doc_name": "test.pdf", "page": 3,
     "text": "Students must register their address within 14 days of moving."},
    {"chunk_id": 1, "doc_name": "test.pdf", "page": 5,
     "text": "The library is open Monday to Friday from 8:00 to 18:00."},
    {"chunk_id": 2, "doc_name": "test.pdf", "page": 8,
     "text": "Exam registration closes four weeks before the exam date."},
]


def fake_embed(texts, model_name=None, is_query=False, batch_size=32):
    """Bag-of-words vectors: texts sharing words get similar vectors."""
    vectors = np.zeros((len(texts), DIM), dtype="float32")
    for row, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            vectors[row, zlib.crc32(word.encode()) % DIM] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "embed_texts", fake_embed)
    retriever.build_from_chunks(CHUNKS, tmp_path, model_name="fake-model")
    return tmp_path


def test_retrieve_returns_best_matching_page(index_dir):
    results = retriever.retrieve("When is the library open?", top_k=1, index_dir=index_dir)
    assert len(results) == 1
    assert results[0]["page"] == 5


def test_retrieve_returns_agreed_chunk_format(index_dir):
    results = retriever.retrieve("exam registration", top_k=3, index_dir=index_dir)
    for chunk in results:
        assert {"chunk_id", "doc_name", "page", "text", "score"} <= chunk.keys()
        assert isinstance(chunk["page"], int)


def test_results_are_sorted_best_first(index_dir):
    results = retriever.retrieve("exam registration date", top_k=3, index_dir=index_dir)
    scores = [c["score"] for c in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["page"] == 8


def test_top_k_larger_than_index_returns_all(index_dir):
    assert len(retriever.retrieve("address", top_k=10, index_dir=index_dir)) == len(CHUNKS)


def test_empty_question_returns_nothing(index_dir):
    assert retriever.retrieve("   ", index_dir=index_dir) == []


def test_saved_index_round_trips(index_dir):
    index, chunks, metadata = vector_store.load_index(index_dir)
    assert index.ntotal == len(CHUNKS)
    assert chunks == CHUNKS
    assert metadata["model_name"] == "fake-model"


def test_missing_index_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Build one first"):
        retriever.retrieve("anything", index_dir=tmp_path / "missing")


def test_index_pdf_end_to_end(tmp_path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    monkeypatch.setattr(embeddings, "embed_texts", fake_embed)

    pdf_path = tmp_path / "sample.pdf"
    with pymupdf.open() as doc:
        for text in ["Opening hours of the library.", "Rules for exam registration."]:
            doc.new_page().insert_text((72, 72), text)
        doc.save(pdf_path)

    count = retriever.index_pdf(pdf_path, tmp_path / "index")
    assert count == 2
    top = retriever.retrieve("exam registration", top_k=1, index_dir=tmp_path / "index")[0]
    assert top["page"] == 2
    assert top["doc_name"] == "sample.pdf"
