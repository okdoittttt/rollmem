# API Reference

The full public API of rollmem. Everything documented here is importable from
the top-level `rollmem` package and listed in `__all__`.

This reference is generated from the source docstrings.

## RollingMemory

::: rollmem.RollingMemory
    options:
      inherited_members: true

## AsyncRollingMemory

::: rollmem.AsyncRollingMemory
    options:
      inherited_members: true

## Message

::: rollmem.Message

## ToolCall

::: rollmem.ToolCall

## Callback protocols

rollmem stays provider-agnostic by letting you inject the two behaviours that
would otherwise tie it to an LLM provider:

```python
SummarizeFn = Callable[[str, Sequence[Message]], str]
AsyncSummarizeFn = Callable[[str, Sequence[Message]], Awaitable[str]]
TokenCounter = Callable[[str], int]
```

- **`SummarizeFn`** — `summarize_fn(existing_summary, messages_to_fold)`
  returns the new summary. Called exactly once per prune, with all evicted
  messages — boundary alignment may enlarge the evicted batch, never the
  number of calls.
- **`AsyncSummarizeFn`** — the coroutine form of the same callback, accepted
  by `AsyncRollingMemory` (which also accepts a plain `SummarizeFn`). Passing
  a coroutine function to the synchronous `RollingMemory` raises `TypeError`.
- **`TokenCounter`** — `token_counter(text)` returns the token count of a
  single message's text. Defaults to a word-count estimate. The text passed in
  is `Message.token_text()` — the content plus any tool-call names, arguments,
  and linkage — so tool payloads count toward the budget.

## Role constants

Conventional role strings, used by the `add_*_message` helpers. rollmem does
not enforce them — `add_message` accepts any role string.

| Constant | Value |
|---|---|
| `USER` | `"user"` |
| `ASSISTANT` | `"assistant"` |
| `SYSTEM` | `"system"` |
| `TOOL` | `"tool"` |
