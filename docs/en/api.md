# API Reference

The full public API of rollmem. Everything documented here is importable from
the top-level `rollmem` package and listed in `__all__`.

This reference is generated from the source docstrings.

## RollingMemory

::: rollmem.RollingMemory

## Message

::: rollmem.Message

## ToolCall

::: rollmem.ToolCall

## Callback protocols

rollmem stays provider-agnostic by letting you inject the two behaviours that
would otherwise tie it to an LLM provider:

```python
SummarizeFn = Callable[[str, Sequence[Message]], str]
TokenCounter = Callable[[str], int]
```

- **`SummarizeFn`** — `summarize_fn(existing_summary, messages_to_fold)`
  returns the new summary. Called once per prune with all evicted messages.
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
