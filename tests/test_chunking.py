import pytest

from docucheck.chunking import chunk_pages, split_text
from docucheck.ingest import clean_text


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", chunk_size=100, overlap=10) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert split_text("   ", chunk_size=100, overlap=10) == []


def test_chunks_respect_size_and_do_not_cut_words():
    text = " ".join(f"word{i}" for i in range(200))
    pieces = split_text(text, chunk_size=100, overlap=20)
    assert len(pieces) > 1
    assert all(len(p) <= 100 for p in pieces)
    words = set(text.split())
    for p in pieces:
        assert set(p.split()) <= words  # every token is a whole word


def test_consecutive_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    pieces = split_text(text, chunk_size=100, overlap=30)
    for first, second in zip(pieces, pieces[1:]):
        assert first.split()[-1] in second.split()


def test_every_word_is_covered():
    text = " ".join(f"word{i}" for i in range(500))
    pieces = split_text(text, chunk_size=120, overlap=25)
    covered = {w for p in pieces for w in p.split()}
    assert covered == set(text.split())


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, overlap=10)


def test_chunk_pages_keeps_page_numbers_and_format():
    pages = [
        {"page": 1, "text": "First page text."},
        {"page": 2, "text": ""},
        {"page": 3, "text": "Third page text."},
    ]
    chunks = chunk_pages(pages, "doc.pdf", chunk_size=100, overlap=10)
    assert chunks == [
        {"chunk_id": 0, "doc_name": "doc.pdf", "page": 1, "text": "First page text."},
        {"chunk_id": 1, "doc_name": "doc.pdf", "page": 3, "text": "Third page text."},
    ]


def test_clean_text_joins_hyphenated_words_and_lines():
    raw = "Exam regis-\ntration closes\nfour weeks before.\n\n\nNew   paragraph."
    assert clean_text(raw) == "Exam registration closes four weeks before.\n\nNew paragraph."
