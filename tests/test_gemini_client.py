"""Tests for llm.gemini_client.GeminiClient (Gemini API mocked out)."""

from unittest.mock import MagicMock, patch

import pytest

from llm.gemini_client import GeminiClient
from utils.exceptions import GeminiAPIError


def _make_client() -> GeminiClient:
    with patch("llm.gemini_client.ChatGoogleGenerativeAI"):
        return GeminiClient(api_key="fake-key")


def test_generate_response_returns_content():
    client = _make_client()
    client.llm.invoke = MagicMock(return_value=MagicMock(content="the answer"))
    assert client.generate_response("prompt") == "the answer"


def test_generate_response_wraps_errors():
    client = _make_client()
    client.llm.invoke = MagicMock(side_effect=RuntimeError("boom"))
    with pytest.raises(GeminiAPIError):
        client.generate_response("prompt")


def test_retry_succeeds_after_failure():
    client = _make_client()
    client.llm.invoke = MagicMock(
        side_effect=[RuntimeError("fail"), MagicMock(content="ok")]
    )
    with patch("llm.gemini_client.time.sleep"):
        assert client.generate_response_with_retry("prompt") == "ok"


def test_retry_exhaustion_raises():
    client = _make_client()
    client.llm.invoke = MagicMock(side_effect=RuntimeError("always fails"))
    with patch("llm.gemini_client.time.sleep"):
        with pytest.raises(GeminiAPIError):
            client.generate_response_with_retry("prompt", max_retries=2)
    assert client.llm.invoke.call_count == 3
