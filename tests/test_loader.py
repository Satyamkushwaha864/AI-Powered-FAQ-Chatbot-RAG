"""Tests for ingestion.loader.DocumentLoader."""

import io

import pytest

from ingestion.loader import DocumentLoader
from utils.exceptions import DocumentParseError, UnsupportedFileTypeError


class FakeFile(io.BytesIO):
    """Minimal stand-in for Streamlit's UploadedFile."""

    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name


def test_load_txt_returns_text():
    loader = DocumentLoader()
    f = FakeFile(b"Hello, world!", "notes.txt")
    assert loader.load(f) == "Hello, world!"


def test_load_md_supported():
    loader = DocumentLoader()
    f = FakeFile("# Title\n\nSome FAQ content.".encode(), "faq.md")
    assert "FAQ content" in loader.load(f)


def test_unsupported_extension_raises():
    loader = DocumentLoader()
    f = FakeFile(b"data", "image.png")
    with pytest.raises(UnsupportedFileTypeError):
        loader.load(f)


def test_empty_txt_raises_parse_error():
    loader = DocumentLoader()
    f = FakeFile(b"   ", "empty.txt")
    with pytest.raises(DocumentParseError):
        loader.load(f)


def test_get_extension_lowercase():
    loader = DocumentLoader()
    assert loader._get_extension("Report.PDF") == ".pdf"
