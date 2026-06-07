"""Provider-agnostic message representation."""

from __future__ import annotations

from dataclasses import dataclass

# Conventional roles. rollmem does not enforce these — any string is accepted —
# but these are the values the built-in helpers (add_user_message, etc.) emit.
USER = "user"
AI = "ai"
SYSTEM = "system"


@dataclass(frozen=True)
class Message:
    """A single turn in a conversation.

    Deliberately minimal so rollmem stays free of any LLM-provider schema.
    Adapters (OpenAI, Anthropic, LangChain, ...) convert to/from this type.
    """

    role: str
    content: str

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"
