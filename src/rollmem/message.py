"""Provider-agnostic message representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

# Conventional roles. rollmem does not enforce these — any string is accepted —
# but these are the values the built-in helpers (add_user_message, etc.) emit.
USER = "user"
ASSISTANT = "assistant"
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

    def to_dict(self) -> Dict[str, str]:
        """Return a plain ``dict`` representation of this message.

        Returns:
            A mapping with ``role`` and ``content`` keys, suitable for JSON
            serialization (the caller chooses the serialization format).
        """
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Mapping[str, str]) -> Message:
        """Reconstruct a message from its ``dict`` representation.

        Args:
            data: A mapping with ``role`` and ``content`` keys, as produced by
                :meth:`to_dict`.

        Returns:
            The reconstructed ``Message``.
        """
        return cls(role=data["role"], content=data["content"])
