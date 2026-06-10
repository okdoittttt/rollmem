# How It Works

rollmem keeps two pieces of state:

- **`buffer`** — recent turns, verbatim, as a `list[Message]`.
- **`summary`** — a running summary of everything that has been evicted from
  the buffer, as a plain `str`.

Both are exposed as public attributes on `RollingMemory` and
`AsyncRollingMemory`, so you can always inspect (or even adjust) the live
state.

## The lifecycle of a turn

1. `add_message` (or one of its role-specific wrappers) appends the turn to
   the buffer.
2. After every append, rollmem checks the buffer's total token count against
   `max_tokens`.
3. If the buffer is over budget, the oldest turns are **evicted**: they are
   folded into `summary` via your `summarize_fn`, then removed from the
   buffer. Without a `summarize_fn`, they are simply dropped.
4. `get_messages()` returns the buffer with the summary prepended as a
   `system` turn; `get_context()` is the same thing joined into one
   prompt-ready string.

## What `max_tokens` means

`max_tokens` is the budget for the **verbatim recent-message buffer** — not
the running summary, and not a model's generation `max_tokens` (output
limit). When the buffer exceeds it, the oldest turns are folded into the
summary until the buffer fits again.

The buffer always keeps **at least the most recent unit** (see below), even
if it alone exceeds the budget. The most recent turn is never summarized away.

## Tool-call units

Eviction never operates on fractions of a tool interaction. The buffer is
partitioned into **units** — spans that are evicted together or kept
together:

1. An assistant message carrying `tool_calls` spans from itself through the
   last buffer message whose `tool_call_id` answers one of its calls —
   including any unrelated messages interleaved in between (keeping those
   while evicting the call would still leave orphaned results behind them).
2. A tool-call message whose results are not (yet) in the buffer is a unit by
   itself.
3. A tool result whose call is absent from the buffer (for example, restored
   from a truncated save) is a single-message unit — handled in normal
   oldest-first order, never an error.
4. Every other message is a single-message unit.

This guarantees `get_messages()` never starts with an orphaned tool result —
a sequence most provider APIs reject. For conversations without tool calls,
every message is its own unit and behaviour is unchanged.

## The pruning algorithm

Pruning is deliberately conservative:

1. First, compute how many of the oldest units must go — without touching
   the buffer.
2. Then summarize all evicted messages in a **single** `summarize_fn` call.
3. Only after that call succeeds are the messages removed from the buffer.

This has two practical consequences:

- **One summarizer call per prune**, not one per message — cheap even when
  many turns are evicted at once.
- **Failure safety.** If `summarize_fn` raises, the buffer is untouched: no
  turns are lost, and the exception propagates to you.

## Sync and async {#sync-and-async}

`AsyncRollingMemory` shares all of the above — the same units, the same
conservative pruning, the same serialization format — with `summarize_fn`
allowed to be a coroutine function (a regular function works too). The
async-specific guarantees:

- **Prunes are serialized.** An internal lock ensures only one prune runs at
  a time, so concurrent tasks never double-evict or overwrite each other's
  summary. Messages added while a summarize call is in flight simply land at
  the tail of the buffer and are never lost.
- **Failure safety carries over.** If the summarizer raises — or the task is
  cancelled mid-`await` — the buffer is untouched, exactly as in the sync
  class.
- **`clear()` wins.** Calling `clear()` while a summarize call is in flight
  discards that fold's result instead of resurrecting the cleared
  conversation.
- **Task-safe, not thread-safe.** One instance is safe across asyncio tasks
  on a single event loop, but must not be shared across threads or event
  loops.

## Token counting

Every budget decision goes through `token_counter`, which takes one message's
text and returns an `int`. The text it receives is `Message.token_text()`:
the content plus each tool call's name and arguments and the `tool_call_id`
linkage, so tool payloads count toward the budget. For a message without tool
fields this is exactly its `content`.

The default counter is a crude word count (`len(text.split())`) —
intentionally simple, so the library works out of the box with zero
dependencies. For real budgets, inject a model-accurate counter such as
`tiktoken` (see [Getting Started](getting-started.md#counting-tokens-accurately)).

## Limitations

!!! warning "Lossy by design"
    Older turns are folded into the summary repeatedly, so each pass can blur
    or drop detail (a "telephone game" effect). Keep `max_tokens` large
    enough that anything you can't afford to lose stays in the verbatim
    buffer.

- **The summary is not bounded for you.** `max_tokens` limits only the
  verbatim buffer. rollmem hands your `summarize_fn` the current summary plus
  the evicted turns and stores whatever it returns — keeping the summary
  compact is your callback's job. If it merely concatenates, the summary (and
  thus `get_context()`) grows without limit.
- **Only as accurate as your counter.** The default token counter is a rough
  word count; inject a model-accurate one for real budgets.
- **No language-specific labels.** The summary is prepended as a bare
  `system` turn. If your prompt needs a label like "Conversation so far:",
  add it in your own prompt assembly.
- **In-memory by default.** State lives in memory; use
  [`to_dict()` / `from_dict()`](persistence.md) to persist and restore it.
