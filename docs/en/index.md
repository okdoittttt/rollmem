# rollmem

Standalone, **dependency-free** rolling conversation memory for LLM apps —
a running summary plus a recent-message buffer.

Inspired by LangChain's `ConversationSummaryBufferMemory`, but with no
LangChain (or any other) dependency. Handy for **conversation memory**,
**context compression**, and long-chat handling when you only need the
summary-buffer pattern.

## Why rollmem?

`ConversationSummaryBufferMemory` is a great pattern: keep recent turns
verbatim, fold older turns into a running summary so context stays bounded.
But pulling in all of LangChain just for that is heavy.

rollmem extracts the idea into a tiny, provider-agnostic package:

- **Zero runtime dependencies.** Nothing but the Python standard library.
- **Provider-agnostic.** You inject how to summarize (`summarize_fn`) and how
  to count tokens (`token_counter`) — rollmem stays neutral, so it works with
  any LLM provider, or with no LLM at all.
- **Fully typed.** Ships `py.typed`, so your type checker sees every
  annotation.
- **Minimal surface.** One memory class, one message type, two callback
  protocols. No configuration sprawl.

## At a glance

```python
from rollmem import RollingMemory

def summarize(existing_summary, messages):
    # plug in any LLM here; return the new summary string
    folded = " ".join(m.content for m in messages)
    return (existing_summary + " " + folded).strip()

mem = RollingMemory(max_tokens=2000, summarize_fn=summarize)

mem.add_user_message("Hi, I'm planning a trip to Korea.")
mem.add_assistant_message("Great! When are you going?")

print(mem.get_context())    # summary (as a system turn) + recent buffer
```

New turns land in a verbatim buffer. When the buffer exceeds `max_tokens`,
the oldest turns are folded into the running summary — so your prompt context
stays bounded while the gist of the whole conversation is preserved.

## Where to next

- [Getting Started](getting-started.md) — install and write your first memory.
- [How It Works](how-it-works.md) — the buffer, the summary, and the pruning
  algorithm, including its limitations.
- [Persistence](persistence.md) — saving and restoring memory state.
- [API Reference](api.md) — the full public API, generated from the source.

## License

MIT
