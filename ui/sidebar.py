"""Sidebar rendering: document upload, KB status, and controls."""

import streamlit as st

from ingestion.chunker import TextChunker
from ingestion.loader import DocumentLoader
from utils.exceptions import RAGPipelineError
from utils.logger import get_logger

logger = get_logger(__name__)


def render_sidebar() -> None:
    """Render the sidebar and handle document uploads.

    Expects ``st.session_state`` to contain ``vector_store``, ``embedder``
    and ``indexed_documents`` (initialized in ``app.py``).
    """
    with st.sidebar:
        st.markdown("## 🤖 AI FAQ Chatbot")
        st.caption("Intelligent document question answering")
        st.divider()

        _render_uploader()
        _render_indexed_documents()
        _render_controls()


def _render_uploader() -> None:
    """Render the file uploader and run the ingestion pipeline."""
    st.markdown("### 📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload documents to add them to the knowledge base.",
        label_visibility="collapsed",
    )

    for uploaded_file in uploaded_files or []:
        already_indexed = any(
            doc["name"] == uploaded_file.name
            for doc in st.session_state.indexed_documents
        )
        if already_indexed:
            continue
        _ingest_file(uploaded_file)


def _ingest_file(uploaded_file) -> None:
    """Run load → chunk → embed → store for a single uploaded file."""
    loader = DocumentLoader()
    chunker = TextChunker(
        chunk_size=st.session_state.config.CHUNK_SIZE,
        chunk_overlap=st.session_state.config.CHUNK_OVERLAP,
    )

    with st.spinner(f"Indexing {uploaded_file.name}..."):
        try:
            text = loader.load(uploaded_file)
            chunks = chunker.split(text, uploaded_file.name)
            if not chunks:
                st.warning(f"⚠️ No usable text found in '{uploaded_file.name}'.")
                return

            embeddings = st.session_state.embedder.embed_texts(
                [c["text"] for c in chunks]
            )
            st.session_state.vector_store.add_documents(embeddings, chunks)
            st.session_state.indexed_documents.append(
                {"name": uploaded_file.name, "chunk_count": len(chunks)}
            )
            st.success(f"✅ {uploaded_file.name} indexed — {len(chunks)} chunks")
        except RAGPipelineError as e:
            logger.exception("Ingestion failed for %s", uploaded_file.name)
            st.error(f"⚠️ Couldn't process '{uploaded_file.name}': {e}")
        except Exception as e:
            logger.exception("Unexpected ingestion error for %s", uploaded_file.name)
            st.error(f"⚠️ Couldn't process '{uploaded_file.name}' — unexpected error.")


def _render_indexed_documents() -> None:
    """Show the list of indexed documents with chunk counts."""
    docs = st.session_state.indexed_documents
    if not docs:
        st.info("No documents indexed yet. Upload a file to get started.")
        return

    st.markdown("### 📚 Knowledge Base")
    for doc in docs:
        st.markdown(f"- ✅ **{doc['name']}** ({doc['chunk_count']} chunks)")


def _render_controls() -> None:
    """Render clear-KB / clear-chat buttons and the settings panel."""
    st.divider()
    st.markdown("### ⚙️ Controls")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear KB", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.indexed_documents = []
            st.rerun()
    with col2:
        if st.button("💬 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    with st.expander("Advanced Settings"):
        config = st.session_state.config
        st.session_state.top_k = st.slider(
            "Top-K results", min_value=1, max_value=10, value=st.session_state.top_k
        )
        st.session_state.show_sources = st.toggle(
            "Show source citations", value=st.session_state.show_sources
        )
        st.caption(
            f"Model: {config.GEMINI_MODEL} · Embeddings: {config.EMBEDDING_MODEL}"
        )

    st.divider()
    st.caption("Powered by Google Gemini • LangChain • FAISS")
