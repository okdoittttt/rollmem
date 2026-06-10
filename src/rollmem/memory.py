"""Core rolling memory: a running summary plus a recent-message buffer.

The behaviour mirrors LangChain's ConversationSummaryBufferMemory but with zero
dependencies. The two things that *would* tie us to an LLM provider — turning
messages into a summary, and counting tokens — are injected by the caller:

    summarize_fn(existing_summary, messages_to_fold) -> new_summary
    token_counter(text) -> int

This keeps rollmem usable with any model, or with no model at all (e.g. a fake
counter and a no-op summarizer in tests).

Two public classes share the same state, semantics, and serialization format:
:class:`RollingMemory` for synchronous code, and :class:`AsyncRollingMemory`
for asyncio code, where ``summarize_fn`` may be a coroutine function.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .message import ASSISTANT, SYSTEM, TOOL, USER, Message, ToolCall

SummarizeFn = Callable[[str, Sequence[Message]], str]
AsyncSummarizeFn = Callable[[str, Sequence[Message]], Awaitable[str]]
TokenCounter = Callable[[str], int]

SCHEMA_VERSION = 2
_READABLE_VERSIONS = (1, 2)


def _default_token_counter(text: str) -> int:
    """Rough word-based estimate used when the caller injects nothing.

    Intentionally crude — real deployments should pass a model-accurate counter
    (e.g. tiktoken). Good enough to make the buffer roll in tests and demos.
    """
    return len(text.split())


def _load_state(memory: _RollingMemoryBase, data: Mapping[str, Any]) -> None:
    """Validate the schema version of ``data`` and restore summary and buffer.

    Args:
        memory: The freshly constructed memory instance to fill.
        data: A mapping produced by ``to_dict``.

    Raises:
        ValueError: If ``data`` has an unsupported serialization version.
    """
    version = data.get("version")
    if version not in _READABLE_VERSIONS:
        raise ValueError(f"unsupported serialization version: {version!r}")
    memory.summary = data.get("summary", "")
    memory.buffer = [Message.from_dict(m) for m in data.get("buffer", [])]


class _RollingMemoryBase:
    """State and pure logic shared by the sync and async memory classes.

    Holds the running summary and the verbatim buffer, the read and
    serialization methods, and the eviction arithmetic. Subclasses add the one
    behaviour that differs: invoking ``summarize_fn`` synchronously or
    asynchronously.
    """

    def __init__(
        self,
        max_tokens: int,
        token_counter: Optional[TokenCounter],
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self._token_counter = token_counter or _default_token_counter
        self.summary: str = ""
        self.buffer: List[Message] = []
        self._epoch = 0

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
        """Drop the summary and the buffer, discarding any in-flight fold."""
        self.summary = ""
        self.buffer.clear()
        self._epoch += 1

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the memory state to a plain ``dict``.

        Only conversation state is captured — the running summary and the
        verbatim buffer. The token budget and the injected callbacks are
        considered runtime configuration, not state, so they are not included
        and must be supplied again at ``from_dict`` time. The format is shared
        by :class:`RollingMemory` and :class:`AsyncRollingMemory`, so state
        saved by one loads in the other.

        Returns:
            A mapping with ``version``, ``summary``, and ``buffer`` keys,
            suitable for JSON serialization (the caller chooses the format).
        """
        return {
            "version": SCHEMA_VERSION,
            "summary": self.summary,
            "buffer": [m.to_dict() for m in self.buffer],
        }

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

    def _eviction_cutoff(self) -> int:
        """Compute how much of the buffer must go for it to fit the budget.

        Eviction operates on the atomic units of :meth:`_units`, never on
        fractions of one, so the buffer never starts with an orphaned tool
        result. At least the most recent unit is always kept, even when it
        alone exceeds the budget.

        Returns:
            The buffer index up to which (exclusive) messages should be
            evicted; ``0`` when nothing needs to be evicted.
        """
        if not self.buffer:
            return 0

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
            return 0
        return units[evict_count - 1][1] + 1


class RollingMemory(_RollingMemoryBase):
    """Keeps recent turns verbatim and folds older turns into a running summary.

    The token budget applies only to the verbatim buffer. The running summary
    is whatever ``summarize_fn`` returns and is not bounded here, so keeping it
    compact is the caller's responsibility: a ``summarize_fn`` that compresses
    keeps ``get_context()`` bounded, while one that merely concatenates lets the
    summary grow without limit.

    This class is synchronous; for asyncio applications — or whenever the
    summarizer is a coroutine function — use :class:`AsyncRollingMemory`.
    Both classes share the same semantics and serialization format.

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
            Must be a regular function — passing a coroutine function raises
            ``TypeError``; use :class:`AsyncRollingMemory` instead.
        token_counter: callback returning a token count for a string. Defaults
            to a word-count estimate.

    Raises:
        TypeError: If ``summarize_fn`` is a coroutine function.
        ValueError: If ``max_tokens`` is not positive.
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        summarize_fn: Optional[SummarizeFn] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        if inspect.iscoroutinefunction(summarize_fn):
            raise TypeError(
                "summarize_fn is a coroutine function; "
                "use AsyncRollingMemory for async summarizers"
            )
        super().__init__(max_tokens, token_counter)
        self._summarize_fn = summarize_fn

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

    # -- serialization ----------------------------------------------------

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
        memory = cls(
            max_tokens=max_tokens,
            summarize_fn=summarize_fn,
            token_counter=token_counter,
        )
        _load_state(memory, data)
        return memory

    # -- internals --------------------------------------------------------

    def _prune(self) -> None:
        """Fold the oldest units into the summary until the buffer fits budget.

        The eviction cutoff is computed first (see :meth:`_eviction_cutoff`),
        the evicted messages are summarized in a single ``summarize_fn`` call,
        and only after that succeeds are they dropped from the buffer. This
        keeps the summarizer call cheap (one call, not one per unit) and means
        a summarizer failure leaves the buffer untouched rather than silently
        losing turns.

        Raises:
            TypeError: If ``summarize_fn`` returns an awaitable — an async
                summarizer was injected into the synchronous class; use
                :class:`AsyncRollingMemory` instead. The buffer is untouched.
        """
        cutoff = self._eviction_cutoff()
        if cutoff == 0:
            return

        evicted = self.buffer[:cutoff]
        if self._summarize_fn is not None:
            # If this raises, we have not touched the buffer yet — no data loss.
            result = self._summarize_fn(self.summary, evicted)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError(
                    "summarize_fn returned an awaitable; "
                    "use AsyncRollingMemory for async summarizers"
                )
            self.summary = result

        del self.buffer[:cutoff]


class AsyncRollingMemory(_RollingMemoryBase):
    """Asyncio variant of :class:`RollingMemory`.

    Behaves identically to the synchronous class — same tool-call eviction
    units, same single-call summarize semantics, same serialization format
    (state saved by one class loads in the other) — except that the ``add_*``
    methods are coroutines and ``summarize_fn`` may be either a regular
    function or a coroutine function; an awaitable result is awaited.

    Safe for concurrent use by multiple asyncio tasks on a single event loop:
    prunes are serialized by an internal lock, a summarizer failure or task
    cancellation leaves the buffer untouched, and messages appended while a
    summarize call is in flight are never lost. Calling :meth:`clear` while a
    summarize call is in flight discards that fold's result — the clear wins.
    Instances are not thread-safe, and one instance must stay on one event
    loop (the internal lock binds to the loop it is first used on).

    Args:
        max_tokens: token budget for the verbatim buffer, exactly as in
            :class:`RollingMemory`.
        summarize_fn: callback that produces an updated summary from the
            current summary plus the messages being evicted. May be a regular
            function or a coroutine function. Required to actually summarize;
            if omitted, evicted messages are dropped (buffer-only). It should
            compress, not just append, to keep the summary bounded.
        token_counter: callback returning a token count for a string. Always
            called synchronously. Defaults to a word-count estimate.

    Raises:
        ValueError: If ``max_tokens`` is not positive.
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        summarize_fn: Optional[Union[SummarizeFn, AsyncSummarizeFn]] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        super().__init__(max_tokens, token_counter)
        self._summarize_fn = summarize_fn
        self._lock: Optional[asyncio.Lock] = None

    # -- adding turns -----------------------------------------------------

    async def add_message(
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
        await self._prune()

    async def add_user_message(self, content: str) -> None:
        await self.add_message(USER, content)

    async def add_assistant_message(self, content: str) -> None:
        await self.add_message(ASSISTANT, content)

    async def add_system_message(self, content: str) -> None:
        await self.add_message(SYSTEM, content)

    async def add_tool_message(
        self, content: str, *, tool_call_id: Optional[str] = None
    ) -> None:
        await self.add_message(TOOL, content, tool_call_id=tool_call_id)

    # -- serialization ----------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        max_tokens: int = 2000,
        summarize_fn: Optional[Union[SummarizeFn, AsyncSummarizeFn]] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> AsyncRollingMemory:
        """Reconstruct a memory from its ``dict`` representation.

        Identical to :meth:`RollingMemory.from_dict` — the serialization format
        is shared, so state saved by either class loads here. The buffer is
        restored verbatim and the pruning logic is not run; the token budget is
        re-applied on the next ``add_message``.

        Args:
            data: A mapping produced by ``to_dict`` of either memory class.
            max_tokens: Token budget for the restored buffer. Runtime
                configuration, not part of the saved state.
            summarize_fn: Summarizer callback to re-inject — regular or
                coroutine function. Callbacks are not serialized, so pass it
                again to keep summarization working.
            token_counter: Token-counter callback to re-inject. Defaults to the
                word-count estimate when omitted.

        Returns:
            The reconstructed ``AsyncRollingMemory``.

        Raises:
            ValueError: If ``data`` has an unsupported serialization version.
        """
        memory = cls(
            max_tokens=max_tokens,
            summarize_fn=summarize_fn,
            token_counter=token_counter,
        )
        _load_state(memory, data)
        return memory

    # -- internals --------------------------------------------------------

    async def _prune(self) -> None:
        """Fold the oldest units into the summary until the buffer fits budget.

        Same contract as :meth:`RollingMemory._prune` — at most one
        ``summarize_fn`` call per prune, and the buffer is only modified after
        that call succeeds — extended for concurrency:

        * Prunes are serialized by a lock; the cutoff is recomputed after
          acquiring it, so a prune that arrives second never re-evicts what an
          earlier one already folded.
        * The current summary is read right before invoking ``summarize_fn``,
          so chained prunes never overwrite each other's result.
        * Messages appended while the summarize call is in flight only land at
          the tail of the buffer, so dropping the evicted prefix afterwards is
          still exact.
        * If :meth:`clear` ran while the summarize call was in flight, the
          fold's result is discarded — the clear wins.
        """
        # Created lazily inside a coroutine: on Python 3.9 a Lock constructed
        # outside a running loop binds to the wrong (implicit) event loop.
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            cutoff = self._eviction_cutoff()
            if cutoff == 0:
                return

            evicted = self.buffer[:cutoff]
            if self._summarize_fn is None:
                del self.buffer[:cutoff]
                return

            epoch = self._epoch
            # If this raises (or the task is cancelled), we have not touched
            # the buffer yet — no data loss.
            result = self._summarize_fn(self.summary, evicted)
            if inspect.isawaitable(result):
                result = await result
            if epoch != self._epoch:
                return

            self.summary = result
            del self.buffer[:cutoff]
