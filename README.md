# rollmem

[![PyPI version](https://img.shields.io/pypi/v/rollmem.svg)](https://pypi.org/project/rollmem/)
[![Python versions](https://img.shields.io/pypi/pyversions/rollmem.svg)](https://pypi.org/project/rollmem/)
[![Typed](https://img.shields.io/badge/typed-py.typed-blue.svg)](https://peps.python.org/pep-0561/)
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

Requires Python 3.9+. Zero runtime dependencies, and fully typed (ships
`py.typed`, so your type checker sees the annotations).

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
mem.add_assistant_message("Great! When are you going?")
mem.add_message("tool", "weather: sunny")   # any role string works

print(mem.get_context())    # -> str: summary (as a system turn) + buffer, joined
print(mem.get_messages())   # -> list[Message]: summary prepended as a system turn
```

`max_tokens` is the budget for the **verbatim recent-message buffer** — not the
running summary, and not a model's generation `max_tokens` (output limit). When
the buffer exceeds it, the oldest turns are folded into the summary.

`token_counter` takes a single message's text (`str`) and returns an `int`. The
default is a crude word count — fine for demos, but pass a model-accurate counter
(such as `tiktoken`) for real token budgets.

## Persistence

`to_dict()` / `from_dict()` serialize the memory **state** (running summary plus
buffer) to and from a plain `dict` — you choose the storage format:

```python
import json

raw = json.dumps(mem.to_dict())   # save anywhere: file, DB column, cache...

mem = RollingMemory.from_dict(
    json.loads(raw),
    max_tokens=2000,
    summarize_fn=summarize,        # callbacks are NOT serialized — re-inject them
    # token_counter=...
)
```

`max_tokens` and the callbacks are runtime configuration, not saved state, so you
pass them again on restore. The buffer is restored verbatim; the token budget is
re-applied on the next added message.

## How it works

- New turns go into `buffer`.
- When `buffer` exceeds `max_tokens`, the oldest turns are folded into `summary`
  via `summarize_fn` (or dropped if none is provided).
- `get_messages() -> list[Message]` returns the buffer with the summary
  prepended as a `system` turn. `get_context() -> str` is the string form of
  the same thing (prompt-ready), so the two never diverge. Neither adds a
  language-specific label — relabel the summary in your own prompt assembly if
  you need to.

## API

`RollingMemory(max_tokens=2000, summarize_fn=None, token_counter=None)`

- `add_message(role, content)` — append a turn with **any** role string.
- `add_user_message` / `add_assistant_message` / `add_system_message` —
  convenience wrappers over `add_message` using the `USER` / `ASSISTANT` /
  `SYSTEM` role constants.
- `get_messages() -> list[Message]` / `get_context() -> str` — read the state
  back (see [How it works](#how-it-works)).
- `to_dict()` / `from_dict(data, *, max_tokens=..., summarize_fn=..., token_counter=...)`
  — serialize and restore (see [Persistence](#persistence)).
- `clear()` — reset the summary and buffer.
- `summary: str` and `buffer: list[Message]` — the live state, exposed as plain
  public attributes.

`Message(role, content)` is the provider-neutral turn type: a frozen dataclass
with `to_dict()` / `from_dict()` and a `"role: content"` string form. The
exported role constants are `USER`, `ASSISTANT`, and `SYSTEM` — but any string
is accepted as a role.

## Limitations

- **Lossy by design.** Older turns are folded into the summary repeatedly, so
  each pass can blur or drop detail (a "telephone game" effect). Keep
  `max_tokens` large enough that anything you can't afford to lose stays in the
  verbatim buffer.
- **The summary is not bounded for you.** `max_tokens` limits only the verbatim
  buffer, not the running summary. rollmem hands your `summarize_fn` the current
  summary plus the evicted turns and stores whatever it returns — so keeping the
  summary compact is your `summarize_fn`'s job. If it merely concatenates,
  the summary (and thus `get_context()`) grows without limit. Prompt it to
  compress, or cap the summary length inside the callback.
- **Only as accurate as your counter.** The default token counter is a rough
  word count; inject a model-accurate one (e.g. `tiktoken`) for real budgets.
- **In-memory by default.** State lives in memory, but `to_dict()` / `from_dict()`
  let you persist and restore it (see [Persistence](#persistence)). Callbacks are
  not serialized and must be re-injected on restore.

## Development

```bash
pip install -e ".[dev]"   # editable install with dev tools (pytest, build, twine)
pytest                    # run the test suite
```

## License

MIT
