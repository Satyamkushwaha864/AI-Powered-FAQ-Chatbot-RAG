"""FAISS-backed vector store with a parallel metadata store."""

import json
from pathlib import Path

import faiss
import numpy as np

from utils.exceptions import VectorStoreError
from utils.logger import get_logger

logger = get_logger(__name__)


class FaissVectorStore:
    """Manages a FAISS ``IndexFlatL2`` index plus chunk metadata.

    FAISS stores only vectors and integer IDs, so a parallel dictionary
    maps each vector ID to its full chunk record (text, source, etc.).
    """

    def __init__(self, dimension: int = 384):
        """Initialize an empty index.

        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata_store: dict[int, dict] = {}
        self._next_id = 0

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self.index.ntotal

    def add_documents(
        self, embeddings: list[list[float]], metadata: list[dict]
    ) -> None:
        """Insert vectors and their metadata into the index.

        Args:
            embeddings: Embedding vectors, one per chunk.
            metadata: Chunk metadata dicts, aligned with ``embeddings``.

        Raises:
            VectorStoreError: If embeddings and metadata lengths differ.
        """
        if len(embeddings) != len(metadata):
            raise VectorStoreError("Embeddings and metadata length mismatch.")
        if not embeddings:
            return

        vectors = np.array(embeddings, dtype="float32")
        if vectors.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension {vectors.shape[1]} does not match "
                f"index dimension {self.dimension}."
            )

        self.index.add(vectors)
        for meta in metadata:
            self.metadata_store[self._next_id] = meta
            self._next_id += 1
        logger.info("Added %d vectors (total: %d)", len(metadata), self.size)

    def search(self, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        """Return the top-k most similar chunks.

        Args:
            query_embedding: The embedded user query.
            top_k: Number of results to return.

        Returns:
            A list of chunk dicts (metadata plus a ``score`` key holding
            the L2 distance), ordered by ascending distance.
        """
        if self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)
        query_vector = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata_store.get(int(idx), {})
            results.append({**meta, "score": float(dist)})
        return results

    def save(self, path: str) -> None:
        """Persist the FAISS index and metadata to disk.

        Args:
            path: Base path (without extension); ``.index`` and
                ``.metadata.json`` are appended.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {"next_id": self._next_id, "metadata": self.metadata_store}, f
            )
        logger.info("Saved index to %s.index", path)

    def load(self, path: str) -> None:
        """Load a previously saved FAISS index and metadata from disk.

        Args:
            path: Base path used in :meth:`save`.

        Raises:
            VectorStoreError: If the saved files cannot be read.
        """
        try:
            self.index = faiss.read_index(f"{path}.index")
            with open(f"{path}.metadata.json", encoding="utf-8") as f:
                data = json.load(f)
            self._next_id = data["next_id"]
            self.metadata_store = {int(k): v for k, v in data["metadata"].items()}
            logger.info("Loaded index from %s.index (%d vectors)", path, self.size)
        except Exception as e:
            raise VectorStoreError(f"Failed to load index from '{path}': {e}") from e

    def clear(self) -> None:
        """Reset the index and metadata store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_store = {}
        self._next_id = 0
        logger.info("Vector store cleared")
