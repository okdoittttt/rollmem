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

This ensures eviction never splits a tool interaction. For conversations
without tool calls, every message is its own unit. What the buffer is allowed
to *start* with after an eviction is handled by boundary alignment, below.

## The pruning algorithm

Pruning is deliberately conservative:

1. First, compute how many of the oldest units must go for the buffer to fit
   the budget — without touching the buffer.
2. Align that cutoff to a valid opening: if the first surviving message would
   be **response-like** — an assistant turn, a tool turn, or an orphaned tool
   result — extend the eviction over whole units up to the first survivor
   that is not. Histories opening with such a turn are rejected by most
   provider APIs (Anthropic, for example, requires the first message to use
   the `user` role). If no valid boundary exists before the last unit,
   the cutoff from step 1 is kept unchanged — evicting more could not fix
   the opening, so the context is kept instead.
3. Then summarize all evicted messages in a **single** `summarize_fn` call.
4. Only after that call succeeds are the messages removed from the buffer.

This has three practical consequences:

- **One summarizer call per prune**, not one per message — cheap even when
  many turns are evicted at once. Alignment may enlarge the evicted batch,
  never the number of calls.
- **After any eviction, the buffer opens on a valid turn.** It never starts
  with an assistant message, a tool message, or an orphaned tool result,
  except in the degenerate case where every remaining unit is response-like
  (see Limitations).
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
- **Boundary alignment is best-effort.** If every unit beyond the token-based
  cutoff opens with a response-like turn (for example, a long run of
  assistant messages), no extension happens and the buffer may still start
  with one. `system` turns and custom role strings are never treated as
  response-like, so a buffer may open with a mid-conversation `system`
  message — fine for most APIs, but adapters that hoist `system` turns out of
  the message list must handle it themselves.
- **Restored buffers are not realigned.** `from_dict` restores verbatim: a
  saved buffer that starts with an assistant turn or an orphaned tool result
  stays that way until the next eviction triggers (see
  [Persistence](persistence.md)).
- **Only as accurate as your counter.** The default token counter is a rough
  word count; inject a model-accurate one for real budgets.
- **No language-specific labels.** The summary is prepended as a bare
  `system` turn. If your prompt needs a label like "Conversation so far:",
  add it in your own prompt assembly.
- **In-memory by default.** State lives in memory; use
  [`to_dict()` / `from_dict()`](persistence.md) to persist and restore it.
