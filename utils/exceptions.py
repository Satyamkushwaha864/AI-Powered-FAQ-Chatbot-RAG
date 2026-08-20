"""Custom exception hierarchy for the RAG pipeline.

All pipeline-specific errors inherit from :class:`RAGPipelineError` so the
Streamlit UI layer can catch a single base type and render a friendly
message while the full stack trace is logged for debugging.
"""


class RAGPipelineError(Exception):
    """Base exception for all pipeline errors."""


class UnsupportedFileTypeError(RAGPipelineError):
    """Raised when an uploaded file has an unsupported extension."""


class DocumentParseError(RAGPipelineError):
    """Raised when a document cannot be parsed or contains no text."""


class EmbeddingModelError(RAGPipelineError):
    """Raised when the embedding model fails to load or encode."""


class VectorStoreError(RAGPipelineError):
    """Raised on FAISS index errors (dimension mismatch, bad state, etc.)."""


class GeminiAPIError(RAGPipelineError):
    """Raised when a call to the Gemini API fails."""
