"""Single-text inference helper for the sentiment classifier.

Used internally by sentiment_tool.py (via lazy import). Operator should
ensure the trained model exists at SENTIMENT_MODEL_PATH before calling.

Public API:
    predict_sentiment(text: str) -> dict
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch

from config.settings import SENTIMENT_MODEL_PATH

# ---------------------------------------------------------------------------
# Label mapping (mirrors train.py and evaluate.py)
# ---------------------------------------------------------------------------

_ID2LABEL: dict[int, str] = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

# ---------------------------------------------------------------------------
# Module-level singleton (lazy-loaded, thread-safe)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_device: str | None = None
_model_lock = threading.Lock()


def _load_model(model_dir: Path) -> None:
    """Load model and tokenizer into module-level cache. Called once."""
    global _model, _tokenizer, _device
    # Import here to avoid heavy startup cost when module is imported
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    _model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    _model.to(_device)
    _model.eval()


def _get_model(model_dir: Path = SENTIMENT_MODEL_PATH):
    """Return the cached model, loading it on first call."""
    global _model, _tokenizer
    if _model is None:
        with _model_lock:
            # Double-checked locking — another thread may have loaded it
            if _model is None:
                if not Path(model_dir).exists():
                    raise FileNotFoundError(
                        f"Trained model not found at {model_dir}. "
                        "Run `python src/models/train.py` first."
                    )
                _load_model(Path(model_dir))
    return _model, _tokenizer, _device


# ---------------------------------------------------------------------------
# Public inference function
# ---------------------------------------------------------------------------


def predict_sentiment(
    text: str,
    model_dir: Path = SENTIMENT_MODEL_PATH,
    max_length: int = 256,
) -> dict:
    """Classify a single review text into NEGATIVE / NEUTRAL / POSITIVE.

    Args:
        text:       Raw review string (any length — truncated to max_length tokens).
        model_dir:  Path to the saved HuggingFace checkpoint directory.
        max_length: Token budget per input (must match training config).

    Returns:
        {
            "sentiment":   "POSITIVE",          # top predicted class
            "confidence":  0.94,                # max softmax score
            "scores": {
                "NEGATIVE": 0.02,
                "NEUTRAL":  0.04,
                "POSITIVE": 0.94,
            }
        }

    Raises:
        FileNotFoundError: if the model checkpoint does not exist.
        ValueError:        if text is empty or whitespace-only.
    """
    text = text.strip() if text else ""
    if not text:
        raise ValueError("predict_sentiment: text must be a non-empty string.")

    model, tokenizer, device = _get_model(model_dir)

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**encoding).logits  # shape: (1, 3)

    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    pred_id = int(torch.argmax(logits, dim=-1).item())

    return {
        "sentiment": _ID2LABEL[pred_id],
        "confidence": round(probs[pred_id], 6),
        "scores": {
            "NEGATIVE": round(probs[0], 6),
            "NEUTRAL":  round(probs[1], 6),
            "POSITIVE": round(probs[2], 6),
        },
    }


# ---------------------------------------------------------------------------
# Quick manual test (not a unit test — requires a trained model on disk)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sample_text = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "This product exceeded all my expectations. Highly recommended!"
    )
    result = predict_sentiment(sample_text)
    print(result)
