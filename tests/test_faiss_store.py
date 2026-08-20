"""Tests for vectorstore.faiss_store.FaissVectorStore."""

import numpy as np
import pytest

from utils.exceptions import VectorStoreError
from vectorstore.faiss_store import FaissVectorStore


def _random_embeddings(n: int, dim: int = 384) -> list[list[float]]:
    rng = np.random.default_rng(42)
    return rng.random((n, dim), dtype=np.float32).tolist()


def test_add_and_search():
    store = FaissVectorStore(dimension=384)
    embeddings = _random_embeddings(5)
    metadata = [{"text": f"chunk {i}", "source": "doc.pdf"} for i in range(5)]
    store.add_documents(embeddings, metadata)

    assert store.size == 5

    results = store.search(embeddings[0], top_k=3)
    assert len(results) == 3
    # The query vector itself should be the closest match.
    assert results[0]["text"] == "chunk 0"
    assert "score" in results[0]


def test_search_empty_store_returns_empty():
    store = FaissVectorStore(dimension=384)
    assert store.search(_random_embeddings(1)[0]) == []


def test_length_mismatch_raises():
    store = FaissVectorStore(dimension=384)
    with pytest.raises(VectorStoreError):
        store.add_documents(_random_embeddings(2), [{"text": "only one"}])


def test_dimension_mismatch_raises():
    store = FaissVectorStore(dimension=384)
    with pytest.raises(VectorStoreError):
        store.add_documents(_random_embeddings(1, dim=8), [{"text": "x"}])


def test_clear_resets_store():
    store = FaissVectorStore(dimension=384)
    store.add_documents(_random_embeddings(2), [{"text": "a"}, {"text": "b"}])
    store.clear()
    assert store.size == 0
    assert store.search(_random_embeddings(1)[0]) == []


def test_save_and_load_roundtrip(tmp_path):
    store = FaissVectorStore(dimension=384)
    embeddings = _random_embeddings(3)
    metadata = [{"text": f"chunk {i}"} for i in range(3)]
    store.add_documents(embeddings, metadata)

    path = str(tmp_path / "index")
    store.save(path)

    loaded = FaissVectorStore(dimension=384)
    loaded.load(path)
    assert loaded.size == 3
    results = loaded.search(embeddings[2], top_k=1)
    assert results[0]["text"] == "chunk 2"
