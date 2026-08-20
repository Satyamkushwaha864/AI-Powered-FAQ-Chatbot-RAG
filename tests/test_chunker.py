"""Tests for ingestion.chunker.TextChunker."""

from ingestion.chunker import TextChunker


def test_split_returns_chunk_records():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = "Sentence one. " * 30  # ~420 chars -> multiple chunks
    chunks = chunker.split(text, "doc.txt")

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_id"] == f"doc.txt_chunk_{i:03}"
        assert chunk["source"] == "doc.txt"
        assert chunk["chunk_index"] == i
        assert len(chunk["text"]) > 0


def test_split_drops_tiny_trailing_chunks():
    chunker = TextChunker(chunk_size=200, chunk_overlap=0)
    chunks = chunker.split("x" * 150 + "\n\n" + "tiny", "doc.txt")
    assert all(len(c["text"]) >= 30 for c in chunks)


def test_split_empty_text_returns_empty_list():
    chunker = TextChunker()
    assert chunker.split("", "doc.txt") == []
