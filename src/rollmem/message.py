"""Provider-agnostic message representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Conventional roles. rollmem does not enforce these — any string is accepted —
# but these are the values the built-in helpers (add_user_message, etc.) emit.
USER = "user"
ASSISTANT = "assistant"
SYSTEM = "system"
TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by an assistant message.

    Deliberately minimal: the intersection of what major providers carry for a
    tool call. ``arguments`` is kept as text (typically JSON) so the type stays
    hashable and token counting stays a plain string operation; adapters encode
    and decode as needed.
    """

    id: str
    name: str
    arguments: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Return a plain ``dict`` representation of this tool call.

        Returns:
            A mapping with ``id``, ``name``, and ``arguments`` keys, suitable
            for JSON serialization.
        """
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: Mapping[str, str]) -> ToolCall:
        """Reconstruct a tool call from its ``dict`` representation.

        Args:
            data: A mapping with ``id`` and ``name`` keys and an optional
                ``arguments`` key, as produced by :meth:`to_dict`.

        Returns:
            The reconstructed ``ToolCall``.
        """
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data.get("arguments", ""),
        )


@dataclass(frozen=True)
class Message:
    """A single turn in a conversation.

    Deliberately minimal so rollmem stays free of any LLM-provider schema.
    Adapters (OpenAI, Anthropic, LangChain, ...) convert to/from this type.

    Beyond ``role`` and ``content``, a message may carry an ``id``, tool-call
    information for agentic conversations, and opaque ``metadata``:

    * ``tool_calls`` holds the tool invocations an assistant message requested.
    * ``tool_call_id`` links a tool-result message (role ``tool``) back to the
      requesting call.
    * ``metadata`` is never interpreted by rollmem — it is round-tripped
      through serialization verbatim and excluded from hashing and token
      counting. Treat it as immutable after construction.
    """

    role: str
    content: str
    id: Optional[str] = None
    tool_calls: Tuple[ToolCall, ...] = ()
    tool_call_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))

    def __str__(self) -> str:
        text = f"{self.role}: {self.content}"
        if self.tool_calls:
            calls = ", ".join(f"{tc.name}({tc.arguments})" for tc in self.tool_calls)
            text = f"{text} [tool calls: {calls}]"
        return text

    def token_text(self) -> str:
        """Canonical text rendering of this message for token counting.

        Includes the content, each tool call's name and arguments, and the
        ``tool_call_id`` linkage, so a plain-string token counter tracks what a
        provider would actually receive. ``metadata`` is excluded. For a
        message without tool fields this is exactly ``content``.

        Returns:
            The text whose token count represents this message.
        """
        parts: List[str] = []
        if self.content:
            parts.append(self.content)
        for tc in self.tool_calls:
            parts.append(f"{tc.name}({tc.arguments})")
        if self.tool_call_id is not None:
            parts.append(self.tool_call_id)
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain ``dict`` representation of this message.

        Optional fields that hold their defaults are omitted, so a plain
        role/content message serializes to the same shape as before these
        fields existed.

        Returns:
            A mapping with ``role`` and ``content`` keys, plus ``id``,
            ``tool_calls``, ``tool_call_id``, and ``metadata`` when set,
            suitable for JSON serialization (the caller chooses the
            serialization format).
        """
        data: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.id is not None:
            data["id"] = self.id
        if self.tool_calls:
            data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Message:
        """Reconstruct a message from its ``dict`` representation.

        Args:
            data: A mapping with ``role`` and ``content`` keys, as produced by
                :meth:`to_dict`. The optional keys (``id``, ``tool_calls``,
                ``tool_call_id``, ``metadata``) fall back to their defaults
                when absent, so older serialized messages load unchanged.

        Returns:
            The reconstructed ``Message``.
        """
        return cls(
            role=data["role"],
            content=data["content"],
            id=data.get("id"),
            tool_calls=tuple(
                ToolCall.from_dict(tc) for tc in data.get("tool_calls", ())
            ),
            tool_call_id=data.get("tool_call_id"),
            metadata=dict(data.get("metadata", {})),
        )
