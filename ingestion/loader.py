"""Extract raw text from uploaded PDF/TXT files."""

from pypdf import PdfReader

from utils.exceptions import DocumentParseError, UnsupportedFileTypeError
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Loads uploaded files and returns their raw text content."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def load(self, uploaded_file) -> str:
        """Dispatch the uploaded file to the correct parser.

        Args:
            uploaded_file: A Streamlit ``UploadedFile`` (or any file-like
                object with a ``name`` attribute).

        Returns:
            The extracted raw text.

        Raises:
            UnsupportedFileTypeError: If the file extension is unsupported.
            DocumentParseError: If the file cannot be parsed.
        """
        ext = self._get_extension(uploaded_file.name)
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")

        logger.info("Loading document: %s", uploaded_file.name)
        if ext == ".pdf":
            return self._load_pdf(uploaded_file)
        return self._load_txt(uploaded_file)

    def _load_pdf(self, uploaded_file) -> str:
        """Extract text from a PDF file object using pypdf."""
        try:
            reader = PdfReader(uploaded_file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if not text.strip():
                raise DocumentParseError(
                    f"No extractable text found in '{uploaded_file.name}'. "
                    "Scanned/image-only PDFs are not supported."
                )
            return text
        except DocumentParseError:
            raise
        except Exception as e:
            raise DocumentParseError(
                f"Failed to parse PDF '{uploaded_file.name}': {e}"
            ) from e

    def _load_txt(self, uploaded_file) -> str:
        """Read raw text from a plain text / markdown file."""
        try:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        except Exception as e:
            raise DocumentParseError(
                f"Failed to read text file '{uploaded_file.name}': {e}"
            ) from e
        if not text.strip():
            raise DocumentParseError(
                f"'{uploaded_file.name}' appears to be empty."
            )
        return text

    def _get_extension(self, filename: str) -> str:
        """Return the lowercase file extension, including the dot."""
        return "." + filename.lower().rsplit(".", 1)[-1]
