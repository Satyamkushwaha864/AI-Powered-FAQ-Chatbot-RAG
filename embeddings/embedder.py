"""Local embedding generation using Sentence Transformers."""

from sentence_transformers import SentenceTransformer

from utils.exceptions import EmbeddingModelError
from utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """Converts text into dense vector embeddings (384-dim by default)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load the Sentence Transformers model.

        Args:
            model_name: Hugging Face model identifier.

        Raises:
            EmbeddingModelError: If the model cannot be loaded.
        """
        try:
            logger.info("Loading embedding model: %s", model_name)
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbeddingModelError(
                f"Failed to load embedding model '{model_name}': {e}"
            ) from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of chunk texts.

        Args:
            texts: Chunk texts to embed.

        Returns:
            A list of L2-normalized embedding vectors.
        """
        try:
            return self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()
        except Exception as e:
            raise EmbeddingModelError(f"Failed to embed texts: {e}") from e

    def embed_query(self, query: str) -> list[float]:
        """Embed a single user query.

        Args:
            query: The user's question.

        Returns:
            An L2-normalized embedding vector.
        """
        try:
            return self.model.encode(
                query, normalize_embeddings=True
            ).tolist()
        except Exception as e:
            raise EmbeddingModelError(f"Failed to embed query: {e}") from e
