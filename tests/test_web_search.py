"""
Unit tests for src/tools/web_search_tool.py.

All tests mock DDGS — no live network calls.

The @tool decorator wraps the function in a crewai Tool object (not callable
directly). Tests invoke the underlying function via `search_market.func(query)`
so that unittest.mock patches apply normally.
"""

import json
from unittest.mock import MagicMock, patch

from src.tools.web_search_tool import search_market

# Shorthand: the raw Python function behind the Tool wrapper.
# Tool now returns JSON-encoded string (so the manager LLM parses reliably) —
# wrap to decode for assertion-friendly dict access in tests.
_search_raw = search_market.func


def _search(*args, **kwargs):
    result = _search_raw(*args, **kwargs)
    return json.loads(result) if isinstance(result, str) else result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_RESULTS = [
    {"title": "Brand A earbuds", "href": "https://example.com/a", "body": "Great sound quality."},
    {"title": "Brand B earbuds", "href": "https://example.com/b", "body": "Budget option."},
    {"title": "Brand C earbuds", "href": "https://example.com/c", "body": "Premium tier."},
]


# ---------------------------------------------------------------------------
# Test 1 — Success path
# ---------------------------------------------------------------------------

@patch("src.tools.web_search_tool.time.sleep")
@patch("src.tools.web_search_tool.DDGS")
def test_success_returns_three_results(mock_ddgs_cls, mock_sleep):
    """Mock returns 3 raw results; tool maps them and returns result_count=3."""
    instance = MagicMock()
    instance.text.return_value = _FAKE_RESULTS
    mock_ddgs_cls.return_value = instance

    result = _search("wireless earbuds market 2026")

    assert result["result_count"] == 3
    assert len(result["results"]) == 3
    assert result["query"] == "wireless earbuds market 2026"
    assert "error" not in result

    # Verify field mapping
    first = result["results"][0]
    assert first["title"] == "Brand A earbuds"
    assert first["url"] == "https://example.com/a"
    assert first["snippet"] == "Great sound quality."

    # Rate-limit floor sleep must have been called
    mock_sleep.assert_any_call(2.0)


# ---------------------------------------------------------------------------
# Test 2 — RatelimitException once, then succeeds
# ---------------------------------------------------------------------------

@patch("src.tools.web_search_tool.time.sleep")
@patch("src.tools.web_search_tool.DDGS")
def test_ratelimit_once_then_success(mock_ddgs_cls, mock_sleep):
    """First DDGS call raises RatelimitException; second call succeeds."""
    from duckduckgo_search.exceptions import RatelimitException

    instance = MagicMock()
    instance.text.side_effect = [
        RatelimitException("rate limited"),
        _FAKE_RESULTS,
    ]
    mock_ddgs_cls.return_value = instance

    result = _search("argan oil competitors")

    assert result["result_count"] == 3
    assert "error" not in result

    # text() must have been called exactly twice
    assert instance.text.call_count == 2

    # Backoff sleep of 2s must appear in the sleep calls (first backoff delay)
    sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
    assert 2 in sleep_calls, f"Expected backoff sleep of 2s; got sleep calls: {sleep_calls}"


# ---------------------------------------------------------------------------
# Test 3 — Persistent RatelimitException (3 retries exhaust)
# ---------------------------------------------------------------------------

@patch("src.tools.web_search_tool.time.sleep")
@patch("src.tools.web_search_tool.DDGS")
def test_persistent_ratelimit_returns_error_dict(mock_ddgs_cls, mock_sleep):
    """All 3 attempts raise RatelimitException; tool returns error dict, no exception."""
    from duckduckgo_search.exceptions import RatelimitException

    instance = MagicMock()
    instance.text.side_effect = RatelimitException("always rate limited")
    mock_ddgs_cls.return_value = instance

    result = _search("standing desks price range")

    # Must not raise — must return controlled error dict
    assert result["results"] == []
    assert result["result_count"] == 0
    assert result["query"] == "standing desks price range"
    assert "error" in result
    assert len(result["error"]) > 0

    # DDGS.text called 3 times (one per retry slot)
    assert instance.text.call_count == 3

    # All three backoff delays must have been slept: 2, 4, 8
    sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
    for delay in (2, 4, 8):
        assert delay in sleep_calls, (
            f"Expected backoff sleep {delay}s; got sleep calls: {sleep_calls}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Unexpected (non-rate-limit) exception returns error dict
# ---------------------------------------------------------------------------

@patch("src.tools.web_search_tool.time.sleep")
@patch("src.tools.web_search_tool.DDGS")
def test_unexpected_exception_returns_error_dict(mock_ddgs_cls, mock_sleep):
    """Non-RatelimitException is caught; tool returns error dict without raising."""
    instance = MagicMock()
    instance.text.side_effect = ConnectionError("network down")
    mock_ddgs_cls.return_value = instance

    result = _search("some query")

    assert result["results"] == []
    assert result["result_count"] == 0
    assert "error" in result
    assert "network down" in result["error"]

    # Only called once — unexpected errors are not retried
    assert instance.text.call_count == 1


# ---------------------------------------------------------------------------
# Test 5 — Empty result list from DDGS
# ---------------------------------------------------------------------------

@patch("src.tools.web_search_tool.time.sleep")
@patch("src.tools.web_search_tool.DDGS")
def test_empty_ddgs_response(mock_ddgs_cls, mock_sleep):
    """DDGS returns empty list; tool returns result_count=0 without error key."""
    instance = MagicMock()
    instance.text.return_value = []
    mock_ddgs_cls.return_value = instance

    result = _search("very obscure niche xyz")

    assert result["result_count"] == 0
    assert result["results"] == []
    assert "error" not in result
