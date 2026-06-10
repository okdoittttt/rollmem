# Persistence

`RollingMemory` lives in memory, but its state can be serialized to a plain
`dict` and restored later — across processes, sessions, or machines. You
choose the storage format and location (JSON file, database column,
cache, ...).

## Saving

`to_dict()` captures the conversation **state**: the running summary and the
verbatim buffer.

```python
import json

raw = json.dumps(mem.to_dict())
# save `raw` anywhere: file, DB column, Redis, ...
```

The resulting mapping has three keys:

```json
{
  "version": 2,
  "summary": "The user is planning a trip to Korea...",
  "buffer": [
    {"role": "user", "content": "What about the weather in Busan?"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {"id": "c1", "name": "get_weather", "arguments": "{\"city\": \"Busan\"}"}
      ]
    },
    {"role": "tool", "content": "sunny, 23C", "tool_call_id": "c1"}
  ]
}
```

Optional message fields (`id`, `tool_calls`, `tool_call_id`, `metadata`) are
omitted when empty, so a plain role/content message serializes to the same
compact shape as before they existed.

`version` is a schema version for forward compatibility — `from_dict` rejects
data with an unsupported version instead of misreading it. Writes are always
the current version (2); version-1 payloads saved by rollmem 0.0.x remain
readable.

## Restoring

```python
mem = RollingMemory.from_dict(
    json.loads(raw),
    max_tokens=2000,
    summarize_fn=summarize,   # callbacks are NOT serialized — re-inject them
    # token_counter=...
)
```

## What is saved, and what is not

| | Saved by `to_dict()` | Passed again to `from_dict()` |
|---|---|---|
| Running summary | ✅ | |
| Message buffer | ✅ | |
| `max_tokens` | | ✅ |
| `summarize_fn` | | ✅ |
| `token_counter` | | ✅ |

The token budget and the callbacks are **runtime configuration, not state**.
Callables cannot be serialized safely, and the budget may legitimately differ
between runs — so both are supplied again at restore time.

## Restore semantics

`from_dict` restores the buffer **verbatim** and does not run the pruning
logic. This means:

- Loading never triggers an unexpected `summarize_fn` call, and never drops
  turns.
- If you restore with a smaller `max_tokens` than the state was saved with,
  the buffer may momentarily exceed the budget — it is re-applied on the next
  `add_message`.

## Raises

`from_dict` raises `ValueError` if `data` has an unsupported serialization
version.
