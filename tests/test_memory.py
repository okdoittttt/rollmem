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
    mem.add_ai_message("world")    # 5 -> total 10, fits
    assert len(mem.buffer) == 2
    assert mem.summary == ""


def test_eviction_without_summarizer_drops_messages():
    mem = RollingMemory(max_tokens=6, token_counter=char_counter)
    mem.add_user_message("aaaa")  # 4
    mem.add_ai_message("bbbb")    # over budget -> oldest dropped
    assert mem.summary == ""
    assert [m.content for m in mem.buffer] == ["bbbb"]


def test_eviction_folds_into_summary():
    mem = RollingMemory(
        max_tokens=6,
        token_counter=char_counter,
        summarize_fn=fake_summarizer,
    )
    mem.add_user_message("aaaa")
    mem.add_ai_message("bbbb")
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
    mem.add_ai_message("bb")    # 2 -> total 4, fits
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
        mem.add_ai_message("bbbb")  # triggers prune -> summarizer raises

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
    mem.add_ai_message("bbbb")
    ctx = mem.get_context()
    assert "Summary of earlier conversation" in ctx
    assert "aaaa" in ctx
    assert "bbbb" in ctx


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
