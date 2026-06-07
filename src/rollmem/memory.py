"""Core rolling memory: a running summary plus a recent-message buffer.

The behaviour mirrors LangChain's ConversationSummaryBufferMemory but with zero
dependencies. The two things that *would* tie us to an LLM provider — turning
messages into a summary, and counting tokens — are injected by the caller:

    summarize_fn(existing_summary, messages_to_fold) -> new_summary
    token_counter(text) -> int

This keeps rollmem usable with any model, or with no model at all (e.g. a fake
counter and a no-op summarizer in tests).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from .message import AI, SYSTEM, USER, Message

SummarizeFn = Callable[[str, Sequence[Message]], str]
TokenCounter = Callable[[str], int]


def _default_token_counter(text: str) -> int:
    """Rough word-based estimate used when the caller injects nothing.

    Intentionally crude — real deployments should pass a model-accurate counter
    (e.g. tiktoken). Good enough to make the buffer roll in tests and demos.
    """
    return len(text.split())


class RollingMemory:
    """Keeps recent turns verbatim and folds older turns into a running summary.

    Args:
        max_tokens: token budget for the verbatim buffer. When the buffer
            exceeds this, the oldest messages are folded into the summary until
            it fits again.
        summarize_fn: callback that produces an updated summary from the current
            summary plus the messages being evicted. Required to actually
            summarize; if omitted, evicted messages are dropped (buffer-only).
        token_counter: callback returning a token count for a string. Defaults
            to a word-count estimate.
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        summarize_fn: Optional[SummarizeFn] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self._summarize_fn = summarize_fn
        self._token_counter = token_counter or _default_token_counter
        self.summary: str = ""
        self.buffer: List[Message] = []

    # -- adding turns -----------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        self.buffer.append(Message(role=role, content=content))
        self._prune()

    def add_user_message(self, content: str) -> None:
        self.add_message(USER, content)

    def add_ai_message(self, content: str) -> None:
        self.add_message(AI, content)

    def add_system_message(self, content: str) -> None:
        self.add_message(SYSTEM, content)

    # -- reading back -----------------------------------------------------

    def get_context(self) -> str:
        """Summary (if any) followed by the verbatim buffer, as one string."""
        parts: List[str] = []
        if self.summary:
            parts.append(f"Summary of earlier conversation:\n{self.summary}")
        parts.extend(str(m) for m in self.buffer)
        return "\n".join(parts)

    def get_messages(self) -> List[Message]:
        """Buffer messages, with the running summary prepended as a system turn."""
        messages: List[Message] = []
        if self.summary:
            messages.append(Message(role=SYSTEM, content=self.summary))
        messages.extend(self.buffer)
        return messages

    def clear(self) -> None:
        self.summary = ""
        self.buffer.clear()

    # -- internals --------------------------------------------------------

    def _buffer_tokens(self) -> int:
        return sum(self._token_counter(m.content) for m in self.buffer)

    def _prune(self) -> None:
        """Fold oldest messages into the summary until the buffer fits budget.

        Eviction is computed first, then summarized in a single ``summarize_fn``
        call, and only after that succeeds are the messages dropped from the
        buffer. This keeps the summarizer call cheap (one call, not one per
        message) and means a summarizer failure leaves the buffer untouched
        rather than silently losing turns.
        """
        # Figure out how many of the oldest messages must go, without mutating
        # the buffer yet. Always keep at least one message in the buffer.
        tokens = self._buffer_tokens()
        evict_count = 0
        while (
            len(self.buffer) - evict_count > 1
            and tokens > self.max_tokens
        ):
            tokens -= self._token_counter(self.buffer[evict_count].content)
            evict_count += 1

        if evict_count == 0:
            return

        evicted = self.buffer[:evict_count]
        if self._summarize_fn is not None:
            # If this raises, we have not touched the buffer yet — no data loss.
            self.summary = self._summarize_fn(self.summary, evicted)

        del self.buffer[:evict_count]
