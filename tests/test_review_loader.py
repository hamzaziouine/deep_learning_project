"""Unit tests for review_loader_tool."""
import json
import os
from pathlib import Path

import pytest

# Point DATA_PATH at the sample CSV before importing the tool module so the
# module-level constant picks up the correct path.
SAMPLE_CSV = str(
    Path(__file__).resolve().parent.parent.parent
    / "data" / "sample" / "electronics_100.csv"
)
os.environ["DATA_PATH"] = SAMPLE_CSV

# Import after env var is set so DATA_PATH is resolved correctly.
from src.tools.review_loader_tool import load_product_reviews  # noqa: E402


# ---------------------------------------------------------------------------
# Test 1: Load Electronics with max_reviews=5
# ---------------------------------------------------------------------------
def test_load_electronics_max_5():
    result = json.loads(load_product_reviews.run(category="Electronics", max_reviews=5))
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert result["category"] == "Electronics"
    assert len(result["reviews"]) == 5
    # total_available should reflect all Electronics rows in the CSV (96 rows)
    assert result["total_available"] > 5
    # Validate each review has the correct schema and types
    for review in result["reviews"]:
        assert isinstance(review["text"], str) and len(review["text"]) > 0
        assert isinstance(review["rating"], int) and 1 <= review["rating"] <= 5
        assert isinstance(review["helpful_votes"], int) and review["helpful_votes"] >= 0
        assert isinstance(review["verified"], bool)


# ---------------------------------------------------------------------------
# Test 2: Load Beauty (small number of rows in sample CSV)
# ---------------------------------------------------------------------------
def test_load_beauty_small_set():
    result = json.loads(load_product_reviews.run(category="Beauty", max_reviews=50))
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert result["category"] == "Beauty"
    # Sample CSV has 4 Beauty rows
    assert len(result["reviews"]) == 4
    assert result["total_available"] == 4


# ---------------------------------------------------------------------------
# Test 3: Unknown category returns error dict with available list
# ---------------------------------------------------------------------------
def test_unknown_category_returns_error_with_available():
    result = json.loads(load_product_reviews.run(category="Toys", max_reviews=10))
    assert result["category"] == "Toys"
    assert result["reviews"] == []
    assert result["total_available"] == 0
    assert "error" in result
    # Error message must include actual available categories
    assert "Electronics" in result["error"]
    assert "Beauty" in result["error"]
    assert "Available" in result["error"]


# ---------------------------------------------------------------------------
# Test 4: max_reviews=0 returns empty list but total_available > 0
# ---------------------------------------------------------------------------
def test_max_reviews_zero():
    result = json.loads(load_product_reviews.run(category="Electronics", max_reviews=0))
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert result["reviews"] == []
    assert result["total_available"] > 0
    assert result["category"] == "Electronics"
