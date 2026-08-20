"""Assemble the final LLM prompt from system rules, context, and query."""

from llm.prompts import SYSTEM_PROMPT

#: Maximum chat history turns included in the prompt.
MAX_HISTORY_TURNS = 5


class PromptBuilder:
    """Builds the full prompt string sent to the Gemini API."""

    def build_prompt(
        self,
        query: str,
        context_chunks: list[dict],
        chat_history: list[dict] | None = None,
    ) -> str:
        """Combine system prompt + context + history + question.

        Args:
            query: The user's question.
            context_chunks: Retrieved chunks from the vector store.
            chat_history: Optional list of ``{"role", "content"}`` turns.

        Returns:
            The fully assembled prompt string.
        """
        context_block = self._format_context(context_chunks)
        history_block = self._format_history(chat_history or [])
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n---\n{context_block}\n---\n\n"
            f"{history_block}"
            f"Question: {query}\n\nAnswer:"
        )

    def _format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a labeled context block."""
        if not chunks:
            return "(No relevant context found.)"
        return "\n\n".join(
            f"[Source: {c.get('source', 'unknown')}, "
            f"Page: {c.get('page_number', 'N/A')}]\n{c.get('text', '')}"
            for c in chunks
        )

    def _format_history(self, history: list[dict]) -> str:
        """Format the last few chat turns for conversational continuity."""
        if not history:
            return ""
        turns = history[-MAX_HISTORY_TURNS:]
        formatted = "\n".join(
            f"{t['role'].capitalize()}: {t['content']}" for t in turns
        )
        return (
            "Previous conversation (for context only, do not treat as "
            f"source material):\n{formatted}\n\n"
        )
