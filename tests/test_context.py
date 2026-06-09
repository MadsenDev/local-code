from local_code.context import build_context_usage, estimate_tokens


def test_estimate_tokens_uses_conservative_utf8_heuristic():
    assert estimate_tokens("12345678") == 2
    assert estimate_tokens("é") == 1


def test_context_usage_accounts_for_each_source_and_remaining_budget():
    usage = build_context_usage(
        history=[{"role": "user", "content": "hello"}],
        memory="project memory",
        repo="repo map",
        tools="tool definitions",
        other="current prompt",
        limit=100,
    )
    assert usage.conversation > 0
    assert usage.memory > 0
    assert usage.repo > 0
    assert usage.tools > 0
    assert usage.other > 0
    assert usage.total == sum([usage.conversation, usage.memory, usage.repo, usage.tools, usage.other])
    assert usage.remaining == 100 - usage.total
    assert usage.to_dict()["total"] == usage.total
