"""Centralized configuration loader (local env + Streamlit Cloud secrets).

All credentials and tunable parameters are sourced here.
Secrets are never hard-coded into the application.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# -------------------------------------------------------------------
# Project base directory
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


# -------------------------------------------------------------------
# Load local .env file
# -------------------------------------------------------------------

load_dotenv(BASE_DIR / ".env")


def _get_secret(name: str, default=None):
    """Get a value from environment variables or Streamlit secrets.

    Local development:
        .env → os.getenv()

    Streamlit Cloud:
        st.secrets
    """

    # 1. Prefer environment variable
    value = os.getenv(name)

    if value is not None and str(value).strip():
        return value

    # 2. Fall back to Streamlit secrets
    try:
        value = st.secrets.get(name)

        if value is not None and str(value).strip():
            return value

    except Exception:
        # st.secrets may not be available during some local runs
        pass

    # 3. Default
    return default


class Config:
    """Application configuration."""

    # ----------------------------------------------------------------
    # API / LLM
    # ----------------------------------------------------------------

    GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")

    GEMINI_MODEL = _get_secret(
        "GEMINI_MODEL",
        "gemini-3.6-flash",
    )

    # ----------------------------------------------------------------
    # Embeddings
    # ----------------------------------------------------------------

    EMBEDDING_MODEL = _get_secret(
        "EMBEDDING_MODEL",
        "all-MiniLM-L6-v2",
    )

    # ----------------------------------------------------------------
    # RAG parameters
    # ----------------------------------------------------------------

    CHUNK_SIZE = int(
        _get_secret("CHUNK_SIZE", "500") or "500"
    )

    CHUNK_OVERLAP = int(
        _get_secret("CHUNK_OVERLAP", "50") or "50"
    )

    TOP_K = int(
        _get_secret("TOP_K", "4") or "4"
    )

    # ----------------------------------------------------------------
    # Gemini generation parameters
    # ----------------------------------------------------------------

    TEMPERATURE = float(
        _get_secret("TEMPERATURE", "0.3") or "0.3"
    )

    MAX_OUTPUT_TOKENS = int(
        _get_secret("MAX_OUTPUT_TOKENS", "1024") or "1024"
    )

    # ----------------------------------------------------------------
    # FAISS
    # ----------------------------------------------------------------

    FAISS_INDEX_PATH = _get_secret(
        "FAISS_INDEX_PATH",
        str(BASE_DIR / "data" / "faiss_index"),
    )

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------

    @staticmethod
    def validate() -> None:
        """Validate required configuration values.

        Raises:
            EnvironmentError: If GEMINI_API_KEY is missing.
        """

        if not Config.GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not configured. "
                "For local development, add it to .env. "
                "For Streamlit Cloud, add it to App Secrets."
            )