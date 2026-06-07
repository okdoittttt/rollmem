from rollmem import RollingMemory, Message


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
    import pytest

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
    import pytest

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
    assert data["version"] == 1
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
    import pytest

    with pytest.raises(ValueError):
        RollingMemory.from_dict({"version": 999, "summary": "", "buffer": []})
