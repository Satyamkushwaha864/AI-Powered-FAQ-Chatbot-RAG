"""Streamlit entry point for the AI-Powered FAQ Chatbot (RAG).

Wires the UI layer to the ingestion → embedding → vector store → LLM
pipeline and owns all Streamlit session state.
"""
def clean_answer(response) -> str:
    """Convert Gemini/LangChain response objects into clean text."""

    if isinstance(response, str):
        return response

    # List of Gemini content blocks
    if isinstance(response, list):
        text_parts = []

        for item in response:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    text_parts.append(text)

        if text_parts:
            return "\n".join(text_parts)

    # Dictionary-style response
    if isinstance(response, dict):
        text = response.get("text")
        if text:
            return text

        content = response.get("content")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        text_parts.append(text)

            if text_parts:
                return "\n".join(text_parts)

    # LangChain/Gemini object with content
    content = getattr(response, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    text_parts.append(text)

        if text_parts:
            return "\n".join(text_parts)

    # Gemini response with .text
    text = getattr(response, "text", None)

    if text:
        return text

    return str(response)

import streamlit as st

from config import Config
from embeddings.embedder import Embedder
from llm.gemini_client import GeminiClient
from ui.chat_view import render_chat
from ui.sidebar import render_sidebar
from utils.logger import get_logger
from vectorstore.faiss_store import FaissVectorStore

logger = get_logger(__name__)

# Embedding dimension of all-MiniLM-L6-v2.
EMBEDDING_DIMENSION = 384


def initialize_session_state() -> None:
    """Initialize all mutable runtime state (idempotent)."""
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("indexed_documents", [])
    st.session_state.setdefault("config", Config)
    st.session_state.setdefault("top_k", Config.TOP_K)
    st.session_state.setdefault("show_sources", True)

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = FaissVectorStore(
            dimension=EMBEDDING_DIMENSION
        )

    if "embedder" not in st.session_state:
        st.session_state.embedder = Embedder(model_name=Config.EMBEDDING_MODEL)

    if "gemini_client" not in st.session_state:
        if not Config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        st.session_state.gemini_client = GeminiClient(
            api_key=Config.GEMINI_API_KEY,
            model_name=Config.GEMINI_MODEL,
            temperature=Config.TEMPERATURE,
            max_output_tokens=Config.MAX_OUTPUT_TOKENS,
        )

def apply_custom_css() -> None:
    """Apply the app's visual theme."""
    st.markdown(
        """
        <style>
        .stApp { background: #0b0f14; }
        section[data-testid="stSidebar"] {
            background: #11161d;
            border-right: 1px solid #252c35;
        }
        .main-title { font-size: 38px; font-weight: 700; margin-bottom: 5px; }
        .subtitle { color: #9aa4b2; font-size: 16px; margin-bottom: 25px; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="AI-Powered FAQ Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        Config.validate()
    except EnvironmentError as e:
        st.error(f"⚠️ Configuration error: {e}")
        st.stop()

    apply_custom_css()
    initialize_session_state()

    st.markdown(
        '<div class="main-title">🤖 AI-Powered FAQ Chatbot</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Ask questions and get answers grounded in '
        "your uploaded documents.</div>",
        unsafe_allow_html=True,
    )

    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
