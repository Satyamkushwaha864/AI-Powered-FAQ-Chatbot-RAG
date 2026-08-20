"""Tests for llm.prompt_builder.PromptBuilder."""

from llm.prompt_builder import PromptBuilder
from llm.prompts import SYSTEM_PROMPT


def test_prompt_contains_system_context_and_question():
    builder = PromptBuilder()
    chunks = [
        {"source": "handbook.pdf", "page_number": 4, "text": "18 days of leave."}
    ]
    prompt = builder.build_prompt("How much leave?", chunks)

    assert SYSTEM_PROMPT in prompt
    assert "[Source: handbook.pdf, Page: 4]" in prompt
    assert "18 days of leave." in prompt
    assert "Question: How much leave?" in prompt
    assert prompt.endswith("Answer:")


def test_empty_context_shows_placeholder():
    builder = PromptBuilder()
    prompt = builder.build_prompt("Anything?", [])
    assert "(No relevant context found.)" in prompt


def test_chat_history_included_and_truncated():
    builder = PromptBuilder()
    history = [
        {"role": "user", "content": f"q{i}"} for i in range(10)
    ]
    prompt = builder.build_prompt("latest?", [], history)
    # Only the last 5 turns are included.
    assert "q9" in prompt
    assert "q5" in prompt
    assert "q4" not in prompt
    assert "Previous conversation" in prompt


def test_no_history_omits_section():
    builder = PromptBuilder()
    prompt = builder.build_prompt("q", [])
    assert "Previous conversation" not in prompt
