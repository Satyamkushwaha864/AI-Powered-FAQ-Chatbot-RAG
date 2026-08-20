"""Split raw text into overlapping, metadata-tagged chunks."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger import get_logger

logger = get_logger(__name__)

#: Trailing chunks shorter than this are dropped as low-quality.
MIN_CHUNK_LENGTH = 30


class TextChunker:
    """Splits text into overlapping chunks using LangChain's splitter."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the chunker.

        Args:
            chunk_size: Target characters per chunk.
            chunk_overlap: Characters of overlap between consecutive chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def split(self, text: str, source_name: str) -> list[dict]:
        """Split text into chunk records with metadata.

        Args:
            text: Raw document text.
            source_name: Name of the source document (e.g. filename).

        Returns:
            A list of chunk dicts with keys ``chunk_id``, ``source``,
            ``chunk_index`` and ``text``.
        """
        raw_chunks = self.splitter.split_text(text)
        chunks = [
            {
                "chunk_id": f"{source_name}_chunk_{i:03}",
                "source": source_name,
                "chunk_index": i,
                "text": chunk,
            }
            for i, chunk in enumerate(raw_chunks)
            if len(chunk.strip()) >= MIN_CHUNK_LENGTH
        ]
        logger.info("Split '%s' into %d chunks", source_name, len(chunks))
        return chunks
