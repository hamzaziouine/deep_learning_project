"""Unit tests for analyze_sentiment tool. No real model weights required — model
loader is patched with a stub that returns controlled logits."""
import json
import types
from unittest.mock import MagicMock, patch

import torch
import pytest


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

def _make_stub_model(scores: list):
    """Return a mock model whose __call__ yields the given softmax scores as logits.

    We pass raw scores directly and bypass softmax inside the tool by making the
    stub return logits whose softmax equals the desired scores.  The simplest
    approach: return log(scores) as logits so softmax(log(scores)) ≈ scores.
    Because the tool computes softmax internally from logits, we use
    log(scores + eps) to avoid -inf.
    """
    import math

    eps = 1e-9
    log_scores = [math.log(s + eps) for s in scores]
    logit_tensor = torch.tensor([log_scores])  # shape (1, 3)

    model_output = MagicMock()
    model_output.logits = logit_tensor

    stub = MagicMock()
    stub.return_value = model_output
    stub.eval = MagicMock()
    return stub


def _make_stub_tokenizer():
    """Return a mock tokenizer that returns minimal valid inputs."""
    stub = MagicMock()
    # __call__ returns a dict-like object accepted by **inputs
    stub.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    return stub


def _patch_load_model(scores: list):
    """Context manager that patches _load_model to return stubs for the given scores."""
    tokenizer = _make_stub_tokenizer()
    model = _make_stub_model(scores)
    # Reset module-level cache so each test gets a fresh load.
    import src.tools.sentiment_tool as module
    return (
        patch.object(module, "_load_model", return_value=(tokenizer, model)),
        patch.object(module, "_tokenizer", None),
        patch.object(module, "_model", None),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnalyzeSentiment:

    def _call(self, text: str, scores: list) -> dict:
        """Helper: patch loader, call tool, return result dict."""
        import src.tools.sentiment_tool as module
        # Reset cache so _load_model is called.
        module._tokenizer = None
        module._model = None

        tokenizer = _make_stub_tokenizer()
        model = _make_stub_model(scores)

        with patch.object(module, "_load_model", return_value=(tokenizer, model)):
            # crewai @tool wraps the function; access the underlying callable via .run
            # or call the function directly by bypassing the decorator.
            result = module.analyze_sentiment.run(review_text=text)
        # Tool now returns JSON-encoded string for reliable LLM parsing — decode for assertions.
        return json.loads(result) if isinstance(result, str) else result

    def test_positive_high_confidence(self):
        """Stub scores [NEG=0.05, NEU=0.05, POS=0.90] → POSITIVE, label_used=POSITIVE."""
        # _LABEL_ORDER = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        result = self._call("This product is absolutely fantastic!", [0.05, 0.05, 0.90])
        assert result["sentiment"] == "POSITIVE"
        assert result["label_used"] == "POSITIVE"
        assert result["confidence"] >= 0.6
        assert "scores" in result
        assert result["scores"]["POSITIVE"] > result["scores"]["NEGATIVE"]

    def test_neutral_low_confidence_uncertain(self):
        """Stub scores [NEG=0.40, NEU=0.45, POS=0.15] → NEUTRAL but confidence 0.45 < 0.6 → UNCERTAIN."""
        result = self._call("It's okay, nothing special.", [0.40, 0.45, 0.15])
        assert result["sentiment"] == "NEUTRAL"
        assert result["label_used"] == "UNCERTAIN"
        assert result["confidence"] < 0.6

    def test_negative_high_confidence(self):
        """Stub scores [NEG=0.85, NEU=0.10, POS=0.05] → NEGATIVE, label_used=NEGATIVE."""
        result = self._call("Terrible product, broke after one day.", [0.85, 0.10, 0.05])
        assert result["sentiment"] == "NEGATIVE"
        assert result["label_used"] == "NEGATIVE"
        assert result["confidence"] >= 0.6
        assert result["scores"]["NEGATIVE"] > result["scores"]["POSITIVE"]

    def test_empty_input_returns_error(self):
        """Empty string → error dict with all scores 0.0, no exception raised."""
        import src.tools.sentiment_tool as module
        # No model patch needed — empty guard fires before model load.
        result = json.loads(module.analyze_sentiment.run(review_text=""))
        assert "error" in result
        assert result["sentiment"] == "UNKNOWN"
        assert result["confidence"] == 0.0
        assert result["label_used"] == "UNCERTAIN"
        assert result["scores"] == {"POSITIVE": 0.0, "NEUTRAL": 0.0, "NEGATIVE": 0.0}

    def test_whitespace_only_input_returns_error(self):
        """Whitespace-only string is treated the same as empty input."""
        import src.tools.sentiment_tool as module
        result = json.loads(module.analyze_sentiment.run(review_text="   \t\n  "))
        assert "error" in result
        assert result["sentiment"] == "UNKNOWN"

    def test_output_schema_keys_present(self):
        """Every successful call must include all four required keys."""
        result = self._call("Good product overall.", [0.10, 0.20, 0.70])
        assert set(result.keys()) >= {"sentiment", "confidence", "label_used", "scores"}
        assert set(result["scores"].keys()) == {"POSITIVE", "NEUTRAL", "NEGATIVE"}

    def test_scores_sum_approximately_one(self):
        """Softmax scores must sum to ~1.0."""
        result = self._call("Average experience.", [0.30, 0.50, 0.20])
        total = sum(result["scores"].values())
        assert abs(total - 1.0) < 0.01
