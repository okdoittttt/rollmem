# Getting Started

## Installation

```bash
pip install rollmem
```

Requires Python 3.9+. There are no runtime dependencies.

## Your first memory

Create a `RollingMemory`, add turns, and read the context back:

```python
from rollmem import RollingMemory

mem = RollingMemory(max_tokens=2000)

mem.add_user_message("Hi, I'm planning a trip to Korea.")
mem.add_assistant_message("Great! When are you going?")

print(mem.get_context())
# user: Hi, I'm planning a trip to Korea.
# assistant: Great! When are you going?
```

`add_user_message`, `add_assistant_message`, `add_system_message`, and
`add_tool_message` are convenience wrappers over `add_message` using the
exported `USER`, `ASSISTANT`, `SYSTEM`, and `TOOL` role constants. Any other
role string is accepted via `add_message`.

Agentic turns carry tool calls and their results:

```python
from rollmem import ASSISTANT, ToolCall

mem.add_message(
    ASSISTANT,
    "",
    tool_calls=[ToolCall(id="c1", name="get_weather", arguments='{"city": "Seoul"}')],
)
mem.add_tool_message("sunny, 23C", tool_call_id="c1")
```

A tool call and its linked results are pruned as one atomic unit, so the
buffer never starts with an orphaned tool result (see
[How It Works](how-it-works.md#tool-call-units)).

Without a `summarize_fn`, rollmem is **buffer-only**: once the buffer exceeds
`max_tokens`, the oldest turns are simply dropped. To preserve their gist,
inject a summarizer.

## Plugging in a summarizer

`summarize_fn` receives the current summary and the messages being evicted,
and returns the new summary. This is where you call your LLM:

```python
from anthropic import Anthropic
from rollmem import RollingMemory

client = Anthropic()

def summarize(existing_summary, messages):
    folded = "\n".join(str(m) for m in messages)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                "Update this running conversation summary with the new turns.\n"
                "Keep it under 200 words; compress, do not just append.\n\n"
                f"Current summary:\n{existing_summary or '(empty)'}\n\n"
                f"New turns:\n{folded}"
            ),
        }],
    )
    return response.content[0].text

mem = RollingMemory(max_tokens=2000, summarize_fn=summarize)
```

!!! tip "Prompt your summarizer to compress"
    rollmem stores whatever your `summarize_fn` returns, without bounding it.
    A summarizer that merely appends makes the summary grow without limit —
    instruct it to compress, or cap its length inside the callback. See
    [How It Works](how-it-works.md#limitations).

## Using it in async applications

In asyncio code, use `AsyncRollingMemory`. It behaves identically — same
eviction units, same serialization format — but the `add_*` methods are
coroutines and `summarize_fn` may be a coroutine function:

```python
from anthropic import AsyncAnthropic
from rollmem import AsyncRollingMemory

client = AsyncAnthropic()

async def summarize(existing_summary, messages):
    folded = "\n".join(str(m) for m in messages)
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                "Update this running conversation summary with the new turns.\n"
                "Keep it under 200 words; compress, do not just append.\n\n"
                f"Current summary:\n{existing_summary or '(empty)'}\n\n"
                f"New turns:\n{folded}"
            ),
        }],
    )
    return response.content[0].text

mem = AsyncRollingMemory(max_tokens=2000, summarize_fn=summarize)

await mem.add_user_message("Hi, I'm planning a trip to Korea.")
```

A regular (synchronous) `summarize_fn` is accepted too, so you can switch
classes without rewriting the callback. The reverse is guarded: passing a
coroutine function to the synchronous `RollingMemory` raises `TypeError`
instead of silently storing a coroutine as the summary. See
[How It Works](how-it-works.md#sync-and-async) for the concurrency
guarantees.

## Counting tokens accurately

The default token counter is a rough word count — fine for demos, but inject
a model-accurate counter for real budgets:

```python
import tiktoken
from rollmem import RollingMemory

enc = tiktoken.get_encoding("cl100k_base")

mem = RollingMemory(
    max_tokens=2000,
    token_counter=lambda text: len(enc.encode(text)),
)
```

`token_counter` takes a single message's text (`str`) and returns an `int`.
The text it receives is `Message.token_text()`, which includes tool-call
names and arguments, so tool payloads count toward the budget.

## Using the memory in a chat loop

`get_messages()` returns the buffer as a `list[Message]` with the running
summary prepended as a `system` turn — ready to convert into your provider's
message format:

```python
history = [
    {"role": m.role, "content": m.content}
    for m in mem.get_messages()
]
```

`get_context()` is the string form of the same thing, so the two never
diverge.

For agentic conversations, map the tool fields (`m.tool_calls`,
`m.tool_call_id`) into your provider's schema as well — the snippet above
covers plain text turns only.
