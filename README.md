# rollmem

[![PyPI version](https://img.shields.io/pypi/v/rollmem.svg)](https://pypi.org/project/rollmem/)
[![Python versions](https://img.shields.io/pypi/pyversions/rollmem.svg)](https://pypi.org/project/rollmem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Standalone, **dependency-free** rolling conversation memory for LLM apps —
a running summary plus a recent-message buffer, inspired by LangChain's
`ConversationSummaryBufferMemory`, but with no LangChain (or any) dependency.

Handy for **conversation memory**, **context compression**, **summarization**,
and **gist**-style long-chat handling — a tiny **LangChain alternative** when
you only need the summary-buffer pattern.

## Why

`ConversationSummaryBufferMemory` is a great pattern: keep recent turns
verbatim, fold older turns into a running summary so context stays bounded.
But pulling in all of LangChain just for that is heavy. `rollmem` extracts the
idea into a tiny, provider-agnostic package. You inject how to summarize and
how to count tokens — rollmem stays neutral.

## Install

```bash
pip install rollmem
```

## Usage

```python
from rollmem import RollingMemory

def summarize(existing_summary, messages):
    # plug in any LLM here; return the new summary string
    folded = " ".join(m.content for m in messages)
    return (existing_summary + " " + folded).strip()

mem = RollingMemory(
    max_tokens=2000,
    summarize_fn=summarize,   # optional; without it, evicted turns are dropped
    # token_counter=...       # optional; defaults to a word-count estimate.
    #                         # In production inject a model-accurate counter, e.g.
    #                         # token_counter=lambda text: len(enc.encode(text))
)

mem.add_user_message("Hi, I'm planning a trip to Korea.")
mem.add_ai_message("Great! When are you going?")

print(mem.get_context())    # -> str: summary (if any) + recent buffer, joined
print(mem.get_messages())   # -> list[Message]: summary prepended as a system turn
```

`token_counter` takes a single message's text (`str`) and returns an `int`. The
default is a crude word count — fine for demos, but pass a model-accurate counter
(such as `tiktoken`) for real token budgets.

## How it works

- New turns go into `buffer`.
- When `buffer` exceeds `max_tokens`, the oldest turns are folded into `summary`
  via `summarize_fn` (or dropped if none is provided).
- `get_context() -> str` returns the summary and buffer joined into one string
  (prompt-ready); `get_messages() -> list[Message]` returns the buffer with the
  summary prepended as a `system` turn.

## Limitations

- **Lossy by design.** Older turns are folded into the summary repeatedly, so
  each pass can blur or drop detail (a "telephone game" effect). Keep
  `max_tokens` large enough that anything you can't afford to lose stays in the
  verbatim buffer.
- **Only as accurate as your counter.** The default token counter is a rough
  word count; inject a model-accurate one (e.g. `tiktoken`) for real budgets.
- **No built-in persistence.** State lives in memory. To save and restore across
  sessions, serialize `summary` and `buffer` yourself.

## License

MIT
