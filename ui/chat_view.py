"""Chat interface rendering: history, input, and answer generation."""

import streamlit as st

from llm.prompt_builder import PromptBuilder
from llm.prompts import NO_CONTEXT_MESSAGE
from utils.exceptions import RAGPipelineError
from utils.logger import get_logger

logger = get_logger(__name__)

WELCOME_MESSAGE = (
    "👋 **Welcome to AI FAQ Assistant**\n\n"
    "Ask questions about the documents in your knowledge base. "
    "I'll retrieve the most relevant information and generate a grounded answer."
)


def render_chat() -> None:
    """Render the chat panel and process new user questions."""
    _render_history()

    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if question and question.strip():
        _handle_question(question.strip())


def _render_history() -> None:
    """Display welcome state and previous chat turns."""

    if not st.session_state.chat_history:
        _render_welcome()
        return

    for turn in st.session_state.chat_history:

        role = turn.get("role", "assistant")
        content = turn.get("content", "")

        avatar = "👤" if role == "user" else "🤖"

        with st.chat_message(role, avatar=avatar):

            st.markdown(content)

            if role == "assistant":
                sources = turn.get("sources", [])

                if sources:
                    _render_sources(sources)


def _render_welcome() -> None:
    """Render the initial empty-chat welcome area."""

    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-icon">🤖</div>
            <div class="welcome-title">
                Welcome to AI FAQ Assistant
            </div>
            <div class="welcome-text">
                Ask questions about your indexed documents.
                The system retrieves relevant information first,
                then uses Google Gemini to generate a grounded answer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 💡 Example questions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("What is the purpose of this project?")

    with col2:
        st.info("What is the technical approach?")

    with col3:
        st.info("What technologies are used?")


def _render_sources(sources: list[str]) -> None:
    """Render source citation information."""

    if not st.session_state.get("show_sources", True):
        return

    unique_sources = sorted(
        {
            str(source).strip()
            for source in sources
            if source and str(source).strip()
        }
    )

    if not unique_sources:
        return

    with st.expander(
        f"📚 Sources ({len(unique_sources)})",
        expanded=False,
    ):
        for source in unique_sources:
            st.markdown(f"- `{source}`")


def _handle_question(question: str) -> None:
    """Run the query pipeline and render the response."""

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):

        if st.session_state.vector_store.size == 0:

            message = (
                "📄 **No document is indexed yet.**\n\n"
                "Please upload and process a document before asking "
                "questions."
            )

            st.markdown(message)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": message,
                    "sources": [],
                }
            )

            return

        with st.spinner("Searching your documents..."):

            try:
                answer, sources = _generate_answer(question)

                if not answer or not answer.strip():
                    answer = (
                        "I couldn't generate a reliable answer from "
                        "the available documents."
                    )

                st.markdown(answer)

                if sources:
                    _render_sources(sources)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except RAGPipelineError as e:

                logger.exception("Query pipeline failed")

                message = (
                    "⚠️ **I couldn't process that question.**\n\n"
                    "Please try again."
                )

                st.error(message)
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": message,
                        "sources": [],
                    }
                )

            except Exception:

                logger.exception("Unexpected query error")

                message = (
                    "⚠️ **Something went wrong.**\n\n"
                    "Please try again."
                )

                st.error(message)
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": message,
                        "sources": [],
                    }
                )


def _generate_answer(question: str) -> tuple[str, list[str]]:
    """Retrieve relevant chunks and generate a grounded answer."""

    query_embedding = (
        st.session_state.embedder.embed_query(question)
    )

    chunks = st.session_state.vector_store.search(
        query_embedding,
        top_k=st.session_state.top_k,
    )

    if not chunks:
        return NO_CONTEXT_MESSAGE, []

    prompt = PromptBuilder().build_prompt(
        query=question,
        context_chunks=chunks,
        chat_history=st.session_state.chat_history,
    )

    answer = (
        st.session_state.gemini_client
        .generate_response_with_retry(prompt)
    )

    sources = [
        c.get("source", "Unknown source")
        for c in chunks
    ]

    return answer, sources