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

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .message import ASSISTANT, SYSTEM, TOOL, USER, Message, ToolCall

SummarizeFn = Callable[[str, Sequence[Message]], str]
TokenCounter = Callable[[str], int]

SCHEMA_VERSION = 2
_READABLE_VERSIONS = (1, 2)


def _default_token_counter(text: str) -> int:
    """Rough word-based estimate used when the caller injects nothing.

    Intentionally crude — real deployments should pass a model-accurate counter
    (e.g. tiktoken). Good enough to make the buffer roll in tests and demos.
    """
    return len(text.split())


class RollingMemory:
    """Keeps recent turns verbatim and folds older turns into a running summary.

    The token budget applies only to the verbatim buffer. The running summary
    is whatever ``summarize_fn`` returns and is not bounded here, so keeping it
    compact is the caller's responsibility: a ``summarize_fn`` that compresses
    keeps ``get_context()`` bounded, while one that merely concatenates lets the
    summary grow without limit.

    Args:
        max_tokens: token budget for the verbatim buffer. When the buffer
            exceeds this, the oldest messages are folded into the summary until
            it fits again. Folding is atomic over tool-call units: an assistant
            message with ``tool_calls`` and its linked tool results are evicted
            together or kept together, so the buffer never starts with an
            orphaned tool result.
            This bounds the buffer only, not the summary, and is
            unrelated to a model's generation ``max_tokens`` (output limit) — it
            is purely the size of the recent-message buffer rollmem keeps.
        summarize_fn: callback that produces an updated summary from the current
            summary plus the messages being evicted. Required to actually
            summarize; if omitted, evicted messages are dropped (buffer-only).
            It should compress, not just append, to keep the summary bounded.
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

    def add_message(
        self,
        role: str,
        content: str,
        *,
        id: Optional[str] = None,
        tool_calls: Sequence[ToolCall] = (),
        tool_call_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.buffer.append(
            Message(
                role=role,
                content=content,
                id=id,
                tool_calls=tuple(tool_calls),
                tool_call_id=tool_call_id,
                metadata=dict(metadata) if metadata else {},
            )
        )
        self._prune()

    def add_user_message(self, content: str) -> None:
        self.add_message(USER, content)

    def add_assistant_message(self, content: str) -> None:
        self.add_message(ASSISTANT, content)

    def add_system_message(self, content: str) -> None:
        self.add_message(SYSTEM, content)

    def add_tool_message(
        self, content: str, *, tool_call_id: Optional[str] = None
    ) -> None:
        self.add_message(TOOL, content, tool_call_id=tool_call_id)

    # -- reading back -----------------------------------------------------

    def get_context(self) -> str:
        """Summary (if any) followed by the verbatim buffer, as one string.

        This is the string form of :meth:`get_messages`: the running summary,
        when present, is rendered as a leading ``system`` turn so both methods
        expose it identically and stay consistent. No language-specific label
        is added — wrap or relabel the summary in your own prompt assembly if
        you need to.
        """
        return "\n".join(str(m) for m in self.get_messages())

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

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the memory state to a plain ``dict``.

        Only conversation state is captured — the running summary and the
        verbatim buffer. The token budget and the injected callbacks are
        considered runtime configuration, not state, so they are not included
        and must be supplied again at :meth:`from_dict` time.

        Returns:
            A mapping with ``version``, ``summary``, and ``buffer`` keys,
            suitable for JSON serialization (the caller chooses the format).
        """
        return {
            "version": SCHEMA_VERSION,
            "summary": self.summary,
            "buffer": [m.to_dict() for m in self.buffer],
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        max_tokens: int = 2000,
        summarize_fn: Optional[SummarizeFn] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> RollingMemory:
        """Reconstruct a memory from its ``dict`` representation.

        The buffer is restored verbatim: this does not call the pruning logic,
        so loading never triggers an unexpected ``summarize_fn`` call or drops
        turns. The token budget is re-applied on the next ``add_message``; if
        ``max_tokens`` is smaller than when the state was saved, the buffer may
        momentarily exceed it until the next turn is added.

        Args:
            data: A mapping produced by :meth:`to_dict`.
            max_tokens: Token budget for the restored buffer. Runtime
                configuration, not part of the saved state.
            summarize_fn: Summarizer callback to re-inject. Callbacks are not
                serialized, so pass it again to keep summarization working.
            token_counter: Token-counter callback to re-inject. Defaults to the
                word-count estimate when omitted.

        Returns:
            The reconstructed ``RollingMemory``.

        Raises:
            ValueError: If ``data`` has an unsupported serialization version.
        """
        version = data.get("version")
        if version not in _READABLE_VERSIONS:
            raise ValueError(f"unsupported serialization version: {version!r}")
        memory = cls(
            max_tokens=max_tokens,
            summarize_fn=summarize_fn,
            token_counter=token_counter,
        )
        memory.summary = data.get("summary", "")
        memory.buffer = [Message.from_dict(m) for m in data.get("buffer", [])]
        return memory

    # -- internals --------------------------------------------------------

    def _buffer_tokens(self) -> int:
        return sum(self._token_counter(m.token_text()) for m in self.buffer)

    def _units(self) -> List[Tuple[int, int]]:
        """Partition the buffer into atomic eviction units.

        A unit is a contiguous ``(start, end)`` span (inclusive indices) that
        must be evicted or kept as a whole, so a tool call and its results are
        never split across the summary/buffer boundary:

        * An assistant message carrying ``tool_calls`` spans from itself
          through the last buffer message whose ``tool_call_id`` answers one of
          its calls — including any unrelated messages interleaved in between.
        * A tool-call message whose results are not (yet) in the buffer, a
          tool result whose call is absent (e.g. restored from a truncated
          save), and every ordinary message each form a single-message unit.
        * Overlapping spans are merged into one unit.

        Returns:
            The unit spans, in order, covering the whole buffer.
        """
        last_result: Dict[str, int] = {}
        for j, message in enumerate(self.buffer):
            if message.tool_call_id is not None:
                last_result[message.tool_call_id] = j

        end_of = list(range(len(self.buffer)))
        for i, message in enumerate(self.buffer):
            if message.tool_calls:
                end_of[i] = max(
                    [i] + [last_result.get(tc.id, i) for tc in message.tool_calls]
                )

        units: List[Tuple[int, int]] = []
        i = 0
        while i < len(self.buffer):
            end = end_of[i]
            j = i
            while j <= end:
                end = max(end, end_of[j])
                j += 1
            units.append((i, end))
            i = end + 1
        return units

    def _prune(self) -> None:
        """Fold the oldest units into the summary until the buffer fits budget.

        Eviction operates on the atomic units of :meth:`_units`, never on
        fractions of one, so the buffer never starts with an orphaned tool
        result. At least the most recent unit is always kept, even when it
        alone exceeds the budget.

        Eviction is computed first, then summarized in a single ``summarize_fn``
        call, and only after that succeeds are the messages dropped from the
        buffer. This keeps the summarizer call cheap (one call, not one per
        unit) and means a summarizer failure leaves the buffer untouched
        rather than silently losing turns.
        """
        if not self.buffer:
            return

        units = self._units()
        unit_tokens = [
            sum(
                self._token_counter(m.token_text())
                for m in self.buffer[start : end + 1]
            )
            for start, end in units
        ]
        total = sum(unit_tokens)
        evict_count = 0
        while len(units) - evict_count > 1 and total > self.max_tokens:
            total -= unit_tokens[evict_count]
            evict_count += 1

        if evict_count == 0:
            return

        cutoff = units[evict_count - 1][1] + 1
        evicted = self.buffer[:cutoff]
        if self._summarize_fn is not None:
            # If this raises, we have not touched the buffer yet — no data loss.
            self.summary = self._summarize_fn(self.summary, evicted)

        del self.buffer[:cutoff]
