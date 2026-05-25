"""Smoke test: verifies the hierarchical crew calls Gemini and returns a string."""
import os

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.smoke
def test_smoke_crew_returns_string():
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set — skipping live Gemini call")

    from src.main import run_smoke_crew

    result = run_smoke_crew()

    assert result is not None, "crew.kickoff() returned None"
    assert str(result).strip() != "", "crew.kickoff() returned an empty string"
