"""Smoke test for `python -m src.main --replay <prefix>`.

This test runs the replay code path against the canonical
demo/wireless_earbuds_demo files. It MUST NOT make any network or LLM calls
— that's the entire point of replay mode (defense-time fallback when Gemini
quota is exhausted).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PREFIX = REPO_ROOT / "demo" / "wireless_earbuds_demo"


def test_replay_prints_known_marker_from_demo(monkeypatch, capsys):
    """--replay must print the saved demo markdown without any LLM call."""
    # Make sure no Gemini key would even be picked up (defensive)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    from src.main import run_replay

    rc = run_replay(str(DEMO_PREFIX))
    assert rc == 0

    out = capsys.readouterr().out
    # Banner must be present so examiners know this is replay, not live.
    assert "REPLAY MODE" in out
    assert "no LLM calls made" in out
    # Known string from demo/wireless_earbuds_demo.md (section header).
    assert "Niche Summary" in out
    # Sentiment table fragment from the demo.
    assert "POSITIVE" in out


def test_replay_missing_file_returns_nonzero(tmp_path, capsys):
    """A missing prefix yields exit-code 1 with a clear stderr message."""
    from src.main import run_replay

    bogus = tmp_path / "does_not_exist"
    rc = run_replay(str(bogus))
    assert rc == 1

    err = capsys.readouterr().err
    assert "replay file not found" in err
