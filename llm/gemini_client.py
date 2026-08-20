"""Google Gemini API client via LangChain, with retry/backoff."""

import time
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from utils.exceptions import GeminiAPIError
from utils.logger import get_logger


logger = get_logger(__name__)


class GeminiClient:
    """Thin wrapper around LangChain's ChatGoogleGenerativeAI."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.6-flash",
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
    ):
        """Initialize the Gemini chat model.

        Args:
            api_key: Google Gemini API key.
            model_name: Gemini model variant.
            temperature: Sampling temperature.
            max_output_tokens: Maximum generated response length.
        """

        if not api_key:
            raise GeminiAPIError("Gemini API key is missing.")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Extract only human-readable text from Gemini content.

        Gemini/LangChain can sometimes return structured content such as:

        [
            {
                "type": "text",
                "text": "Actual answer",
                "extras": {
                    "signature": "..."
                }
            }
        ]

        We only want the 'text' field and never expose
        internal metadata such as signatures.
        """

        # Normal string response
        if isinstance(content, str):
            return content.strip()

        # Structured list response
        if isinstance(content, list):
            text_parts: list[str] = []

            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")

                    if text:
                        text_parts.append(str(text))

                elif isinstance(item, str):
                    text_parts.append(item)

            if text_parts:
                return "\n".join(text_parts).strip()

        # Dictionary response
        if isinstance(content, dict):
            text = content.get("text")

            if text:
                return str(text).strip()

            nested_content = content.get("content")

            if isinstance(nested_content, str):
                return nested_content.strip()

            if isinstance(nested_content, list):
                return GeminiClient._extract_text(nested_content)

        # Some response objects expose a .text property
        text = getattr(content, "text", None)

        if isinstance(text, str) and text.strip():
            return text.strip()

        # Safe fallback
        return str(content).strip()

    def generate_response(self, prompt: str) -> str:
        """Send a prompt to Gemini and return clean answer text.

        Args:
            prompt: The fully assembled prompt string.

        Returns:
            The generated human-readable answer.

        Raises:
            GeminiAPIError: If the API call fails.
        """

        if not prompt or not prompt.strip():
            raise GeminiAPIError("Prompt cannot be empty.")

        try:
            response = self.llm.invoke(prompt)

            # LangChain AIMessage normally exposes the generated
            # response through the 'content' attribute.
            content = response.content

            # Extract only readable text.
            answer = self._extract_text(content)

            if not answer:
                raise GeminiAPIError(
                    "Gemini returned an empty response."
                )

            return answer

        except GeminiAPIError:
            raise

        except Exception as e:
            logger.exception("Gemini API call failed.")
            raise GeminiAPIError(
                f"Gemini API call failed: {e}"
            ) from e

    def generate_response_with_retry(
        self,
        prompt: str,
        max_retries: int = 2,
    ) -> str:
        """Call generate_response with exponential backoff.

        Args:
            prompt: The fully assembled prompt string.
            max_retries: Number of retries after the initial attempt.

        Returns:
            The generated answer.

        Raises:
            GeminiAPIError: If all attempts fail.
        """

        last_error: GeminiAPIError | None = None

        for attempt in range(max_retries + 1):
            try:
                return self.generate_response(prompt)

            except GeminiAPIError as e:
                last_error = e

                if attempt < max_retries:
                    wait = 2 ** attempt

                    logger.warning(
                        "Gemini call failed "
                        "(attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        max_retries + 1,
                        wait,
                        e,
                    )

                    time.sleep(wait)

        if last_error is not None:
            raise last_error

        raise GeminiAPIError(
            "Gemini API request failed for an unknown reason."
        )