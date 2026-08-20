"""Centralized configuration loader (env + defaults).

All credentials and tunable parameters are sourced here — never hard-code
secrets elsewhere in the codebase.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """Application configuration sourced from environment variables."""

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K = int(os.getenv("TOP_K", "4"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
    FAISS_INDEX_PATH = os.getenv(
        "FAISS_INDEX_PATH", str(BASE_DIR / "data" / "faiss_index")
    )

    @staticmethod
    def validate() -> None:
        """Ensure required environment variables are present.

        Raises:
            EnvironmentError: If ``GEMINI_API_KEY`` is not set.
        """
        if not Config.GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and "
                "add your key."
            )
