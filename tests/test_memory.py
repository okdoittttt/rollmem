import pytest

from rollmem import ASSISTANT, TOOL, Message, RollingMemory, ToolCall


def char_counter(text: str) -> int:
    """Deterministic token counter for tests: 1 token per character."""
    return len(text)


def fake_summarizer(existing: str, evicted) -> str:
    folded = " ".join(m.content for m in evicted)
    return (existing + " " + folded).strip()


def test_buffer_only_keeps_recent():
    mem = RollingMemory(max_tokens=10, token_counter=char_counter)
    mem.add_user_message("hello")  # 5
    mem.add_assistant_message("world")    # 5 -> total 10, fits
    assert len(mem.buffer) == 2
    assert mem.summary == ""


def test_eviction_without_summarizer_drops_messages():
    mem = RollingMemory(max_tokens=6, token_counter=char_counter)
    mem.add_user_message("aaaa")  # 4
    mem.add_assistant_message("bbbb")    # over budget -> oldest dropped
    assert mem.summary == ""
    assert [m.content for m in mem.buffer] == ["bbbb"]


def test_eviction_folds_into_summary():
    mem = RollingMemory(
        max_tokens=6,
        token_counter=char_counter,
        summarize_fn=fake_summarizer,
    )
    mem.add_user_message("aaaa")
    mem.add_assistant_message("bbbb")
    assert mem.summary == "aaaa"
    assert [m.content for m in mem.buffer] == ["bbbb"]


def test_multiple_evictions_summarize_in_one_call():
    calls = []

    def counting_summarizer(existing, evicted):
        calls.append(list(evicted))
        return existing + " " + " ".join(m.content for m in evicted)

    mem = RollingMemory(
        max_tokens=4,
        token_counter=char_counter,
        summarize_fn=counting_summarizer,
    )
    mem.add_user_message("aa")  # 2
    mem.add_assistant_message("bb")    # 2 -> total 4, fits
    mem.add_user_message("cccc")  # pushes total to 8; "aa" and "bb" must go

    # Both evicted messages summarized together: exactly one summarizer call.
    assert len(calls) == 1
    assert [m.content for m in calls[0]] == ["aa", "bb"]
    assert [m.content for m in mem.buffer] == ["cccc"]


def test_summarizer_failure_does_not_lose_messages():
    def failing_summarizer(existing, evicted):
        raise RuntimeError("LLM unavailable")

    mem = RollingMemory(
        max_tokens=4,
        token_counter=char_counter,
        summarize_fn=failing_summarizer,
    )
    mem.add_user_message("aaaa")  # 4, fits
    with pytest.raises(RuntimeError):
        mem.add_assistant_message("bbbb")  # triggers prune -> summarizer raises

    # Buffer untouched, nothing summarized: no data loss.
    assert mem.summary == ""
    assert [m.content for m in mem.buffer] == ["aaaa", "bbbb"]


def test_get_context_includes_summary():
    mem = RollingMemory(
        max_tokens=6,
        token_counter=char_counter,
        summarize_fn=fake_summarizer,
    )
    mem.add_user_message("aaaa")
    mem.add_assistant_message("bbbb")
    ctx = mem.get_context()
    # Summary is rendered as a leading system turn, matching get_messages.
    assert "system: aaaa" in ctx
    assert "assistant: bbbb" in ctx
    # get_context is exactly the string form of get_messages.
    assert ctx == "\n".join(str(m) for m in mem.get_messages())


def test_clear():
    mem = RollingMemory(max_tokens=6, token_counter=char_counter)
    mem.add_user_message("x")
    mem.clear()
    assert mem.buffer == []
    assert mem.summary == ""


def test_invalid_max_tokens():
    with pytest.raises(ValueError):
        RollingMemory(max_tokens=0)


def test_message_round_trips():
    msg = Message(role="user", content="hello")
    assert msg.to_dict() == {"role": "user", "content": "hello"}
    assert Message.from_dict(msg.to_dict()) == msg


def test_memory_to_dict_shape():
    mem = RollingMemory(max_tokens=100, token_counter=char_counter)
    mem.add_user_message("hi")
    mem.add_assistant_message("yo")

    data = mem.to_dict()
    assert set(data) == {"version", "summary", "buffer"}
    assert data["version"] == 2
    assert all(isinstance(entry, dict) for entry in data["buffer"])
    assert [entry["content"] for entry in data["buffer"]] == ["hi", "yo"]


def test_memory_round_trips():
    mem = RollingMemory(
        max_tokens=6,
        token_counter=char_counter,
        summarize_fn=fake_summarizer,
    )
    mem.add_user_message("aaaa")
    mem.add_assistant_message("bbbb")  # evicts "aaaa" into the summary
    assert mem.summary == "aaaa"
    assert [m.content for m in mem.buffer] == ["bbbb"]

    restored = RollingMemory.from_dict(
        mem.to_dict(),
        max_tokens=6,
        token_counter=char_counter,
        summarize_fn=fake_summarizer,
    )
    assert restored.summary == mem.summary
    assert [m.content for m in restored.buffer] == [m.content for m in mem.buffer]


def test_from_dict_restores_working_memory():
    saved = {"version": 1, "summary": "earlier", "buffer": [{"role": "user", "content": "bbbb"}]}

    mem = RollingMemory.from_dict(
        saved,
        max_tokens=6,
        token_counter=char_counter,
        summarize_fn=fake_summarizer,
    )
    # Buffer restored verbatim; budget not enforced until the next add.
    assert [m.content for m in mem.buffer] == ["bbbb"]

    # Re-injected callback is live: a new turn triggers eviction + summary.
    mem.add_assistant_message("cccc")
    assert mem.summary == "earlier bbbb"
    assert [m.content for m in mem.buffer] == ["cccc"]


def test_from_dict_rejects_unknown_version():
    with pytest.raises(ValueError):
        RollingMemory.from_dict({"version": 999, "summary": "", "buffer": []})


# -- agentic messages: ToolCall, ids, metadata --------------------------


def test_tool_call_round_trips():
    call = ToolCall(id="c1", name="get_weather", arguments='{"city": "Paris"}')
    assert call.to_dict() == {
        "id": "c1",
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
    }
    assert ToolCall.from_dict(call.to_dict()) == call


def test_message_with_tool_fields_round_trips():
    msg = Message(
        role=ASSISTANT,
        content="checking",
        id="m1",
        tool_calls=(ToolCall(id="c1", name="f", arguments="{}"),),
        metadata={"trace": "abc"},
    )
    restored = Message.from_dict(msg.to_dict())
    assert restored == msg

    result = Message(role=TOOL, content="sunny", tool_call_id="c1")
    assert Message.from_dict(result.to_dict()) == result


def test_plain_message_to_dict_omits_optional_fields():
    msg = Message(role="user", content="hello")
    assert msg.to_dict() == {"role": "user", "content": "hello"}


def test_message_from_dict_accepts_v1_shape():
    msg = Message.from_dict({"role": "user", "content": "hello"})
    assert msg.id is None
    assert msg.tool_calls == ()
    assert msg.tool_call_id is None
    assert msg.metadata == {}


def test_message_hashable_with_metadata():
    a = Message(role="user", content="hi", metadata={"k": 1})
    b = Message(role="user", content="hi", metadata={"k": 1})
    assert hash(a) == hash(b)
    assert a == b
    assert a != Message(role="user", content="hi", metadata={"k": 2})


def test_tool_calls_list_normalized_to_tuple():
    msg = Message(
        role=ASSISTANT,
        content="",
        tool_calls=[ToolCall(id="c1", name="f")],
    )
    assert isinstance(msg.tool_calls, tuple)
    hash(msg)


def test_token_text_includes_tool_calls_and_linkage():
    plain = Message(role="user", content="hello")
    assert plain.token_text() == "hello"

    call = Message(
        role=ASSISTANT,
        content="checking",
        tool_calls=(ToolCall(id="c1", name="f", arguments='{"x": 1}'),),
    )
    assert "f(" in call.token_text()
    assert '{"x": 1}' in call.token_text()

    result = Message(role=TOOL, content="sunny", tool_call_id="c1")
    assert "sunny" in result.token_text()
    assert "c1" in result.token_text()


def test_str_shows_tool_calls():
    call = Message(
        role=ASSISTANT,
        content="",
        tool_calls=(ToolCall(id="c1", name="f", arguments="{}"),),
    )
    assert "f({})" in str(call)

    result = Message(role=TOOL, content="sunny", tool_call_id="c1")
    assert str(result) == "tool: sunny"


def test_add_tool_message_helper():
    mem = RollingMemory(max_tokens=100, token_counter=char_counter)
    mem.add_tool_message("output", tool_call_id="c1")
    assert mem.buffer[0].role == TOOL
    assert mem.buffer[0].tool_call_id == "c1"


# -- agentic messages: unit-atomic eviction -----------------------------


def test_tool_unit_evicted_atomically():
    calls = []

    def counting_summarizer(existing, evicted):
        calls.append(list(evicted))
        return existing + " " + " ".join(m.content for m in evicted)

    mem = RollingMemory(
        max_tokens=20,
        token_counter=char_counter,
        summarize_fn=counting_summarizer,
    )
    mem.add_user_message("aaaa")  # 4
    mem.add_message(
        ASSISTANT, "", tool_calls=[ToolCall(id="c1", name="f", arguments="xx")]
    )  # "f(xx)" -> 5, total 9
    mem.add_tool_message("rrrr", tool_call_id="c1")  # "rrrr\nc1" -> 7, total 16
    mem.add_user_message("z" * 12)  # total 28: evicts user AND the whole pair

    assert len(calls) == 1
    assert [m.role for m in calls[0]] == ["user", ASSISTANT, TOOL]
    assert [m.content for m in mem.buffer] == ["z" * 12]


def test_no_orphaned_tool_result_at_head():
    mem = RollingMemory(
        max_tokens=12,
        token_counter=char_counter,
        summarize_fn=fake_summarizer,
    )
    mem.add_message(
        ASSISTANT, "", tool_calls=[ToolCall(id="c1", name="f", arguments="xx")]
    )  # 5
    mem.add_tool_message("rrrr", tool_call_id="c1")  # 7, total 12, fits
    # Per-message eviction would drop only the call (5 tokens suffice) and
    # leave the tool result orphaned at the head. Units keep the pair together.
    mem.add_user_message("zz")

    assert [m.content for m in mem.buffer] == ["zz"]
    assert mem.buffer[0].tool_call_id is None


def test_last_unit_kept_even_over_budget():
    mem = RollingMemory(
        max_tokens=5,
        token_counter=char_counter,
        summarize_fn=fake_summarizer,
    )
    mem.add_message(
        ASSISTANT, "", tool_calls=[ToolCall(id="c1", name="f", arguments="xxxx")]
    )  # 7 > 5, but it is the only unit
    mem.add_tool_message("rrrr", tool_call_id="c1")  # pair now 14 > 5, still one unit

    assert mem.summary == ""
    assert len(mem.buffer) == 2


def test_oversized_old_unit_evicted_whole():
    calls = []

    def counting_summarizer(existing, evicted):
        calls.append(list(evicted))
        return existing + " " + " ".join(m.content for m in evicted)

    mem = RollingMemory(
        max_tokens=6,
        token_counter=char_counter,
        summarize_fn=counting_summarizer,
    )
    mem.add_message(
        ASSISTANT, "", tool_calls=[ToolCall(id="c1", name="f", arguments="xxxx")]
    )
    mem.add_tool_message("rrrr", tool_call_id="c1")  # unit of 14, kept while newest
    mem.add_user_message("zzzz")  # new last unit: the oversized pair goes, whole

    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert [m.content for m in mem.buffer] == ["zzzz"]


def test_multiple_results_one_call_grouped():
    calls = []

    def counting_summarizer(existing, evicted):
        calls.append(list(evicted))
        return existing

    mem = RollingMemory(
        max_tokens=10,
        token_counter=char_counter,
        summarize_fn=counting_summarizer,
    )
    mem.add_message(
        ASSISTANT,
        "",
        tool_calls=[
            ToolCall(id="c1", name="f", arguments="x"),
            ToolCall(id="c2", name="g", arguments="y"),
        ],
    )
    mem.add_tool_message("r1", tool_call_id="c1")
    mem.add_tool_message("r2", tool_call_id="c2")
    assert calls == []  # call + both results form the newest unit: protected
    mem.add_user_message("z" * 8)

    assert len(calls) == 1
    assert [m.role for m in calls[0]] == [ASSISTANT, TOOL, TOOL]
    assert [m.content for m in mem.buffer] == ["z" * 8]


def test_interleaved_message_inside_unit():
    calls = []

    def counting_summarizer(existing, evicted):
        calls.append(list(evicted))
        return existing

    mem = RollingMemory(
        max_tokens=12,
        token_counter=char_counter,
        summarize_fn=counting_summarizer,
    )
    mem.add_message(
        ASSISTANT, "", tool_calls=[ToolCall(id="c1", name="f", arguments="xx")]
    )
    mem.add_user_message("nn")  # interleaved between call and result
    mem.add_tool_message("rr", tool_call_id="c1")
    mem.add_user_message("zzzz")

    assert len(calls) == 1
    assert [m.content for m in calls[0]] == ["", "nn", "rr"]
    assert [m.content for m in mem.buffer] == ["zzzz"]


def test_orphan_tool_result_is_own_unit():
    saved = {
        "version": 2,
        "summary": "",
        "buffer": [
            {"role": "tool", "content": "rrrr", "tool_call_id": "missing"},
            {"role": "user", "content": "hi"},
        ],
    }
    mem = RollingMemory.from_dict(
        saved,
        max_tokens=6,
        token_counter=char_counter,
        summarize_fn=fake_summarizer,
    )
    mem.add_assistant_message("yo")  # prunes the orphan standalone, no error

    assert "rrrr" in mem.summary
    assert [m.content for m in mem.buffer] == ["hi", "yo"]


def test_token_counting_uses_token_text():
    mem = RollingMemory(
        max_tokens=8,
        token_counter=char_counter,
        summarize_fn=fake_summarizer,
    )
    mem.add_user_message("aaaa")  # 4
    # Empty content but 7 tokens of rendered tool call ("f(xxxx)"). If counting
    # looked at bare content this would weigh 0 and nothing would be evicted.
    mem.add_message(
        ASSISTANT, "", tool_calls=[ToolCall(id="c1", name="f", arguments="xxxx")]
    )

    assert mem.summary == "aaaa"
    assert [m.role for m in mem.buffer] == [ASSISTANT]


def test_summarizer_failure_with_units_loses_nothing():
    def failing_summarizer(existing, evicted):
        raise RuntimeError("LLM unavailable")

    mem = RollingMemory(
        max_tokens=12,
        token_counter=char_counter,
        summarize_fn=failing_summarizer,
    )
    mem.add_message(
        ASSISTANT, "", tool_calls=[ToolCall(id="c1", name="f", arguments="xx")]
    )
    mem.add_tool_message("rrrr", tool_call_id="c1")

    with pytest.raises(RuntimeError):
        mem.add_user_message("zz")

    assert mem.summary == ""
    assert len(mem.buffer) == 3


# -- agentic messages: serialization ------------------------------------


def test_memory_round_trips_with_tool_messages():
    mem = RollingMemory(max_tokens=1000, token_counter=char_counter)
    mem.add_user_message("hi")
    mem.add_message(
        ASSISTANT,
        "checking",
        id="m1",
        tool_calls=[ToolCall(id="c1", name="f", arguments="{}")],
        metadata={"trace": "abc"},
    )
    mem.add_tool_message("sunny", tool_call_id="c1")

    restored = RollingMemory.from_dict(
        mem.to_dict(), max_tokens=1000, token_counter=char_counter
    )
    assert restored.buffer == mem.buffer


def test_from_dict_accepts_version_1_payload():
    saved = {
        "version": 1,
        "summary": "earlier",
        "buffer": [{"role": "user", "content": "hi"}],
    }
    mem = RollingMemory.from_dict(saved, max_tokens=100, token_counter=char_counter)
    assert mem.summary == "earlier"
    assert [m.content for m in mem.buffer] == ["hi"]

    with pytest.raises(ValueError):
        RollingMemory.from_dict({"version": 3, "summary": "", "buffer": []})
