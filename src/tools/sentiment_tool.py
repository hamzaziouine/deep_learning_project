"""Sentiment analysis tool wrapping a fine-tuned RoBERTa 3-class classifier."""
import json
import os

import torch
from crewai.tools import tool

from config import settings

# ---------------------------------------------------------------------------
# Module-level cache — populated on first call (lazy load).
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None
_device = None
_logger = None


def set_logger(logger) -> None:
    """A7 wires the AgentLogger here at crew setup. Optional — tool works without."""
    global _logger
    _logger = logger


def _load_model():
    """Load tokenizer and model from MODEL_PATH; cache at module level. Moves to CUDA when available."""
    global _tokenizer, _model, _device
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = str(os.getenv("MODEL_PATH", default=str(settings.SENTIMENT_MODEL_PATH)))
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _model.to(_device)
    _model.eval()
    return _tokenizer, _model


# Class order must match the label mapping used during training:
# index 0 → NEGATIVE, index 1 → NEUTRAL, index 2 → POSITIVE
_LABEL_ORDER = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("CONFIDENCE_THRESHOLD", default=str(settings.DEFAULT_CONFIDENCE_THRESHOLD))
)

_EMPTY_RESPONSE = {
    "sentiment": "UNKNOWN",
    "confidence": 0.0,
    "label_used": "UNCERTAIN",
    "scores": {"POSITIVE": 0.0, "NEUTRAL": 0.0, "NEGATIVE": 0.0},
    "error": "Empty review text",
}


@tool("analyze_sentiment")
def analyze_sentiment(review_text: str) -> dict:
    """Classify review sentiment (POSITIVE/NEUTRAL/NEGATIVE) using fine-tuned RoBERTa.
    Returns UNCERTAIN if confidence < threshold."""
    if not review_text or not review_text.strip():
        if _logger is not None:
            try:
                _logger.tool_called("analyze_sentiment", {"review_text": ""}, _EMPTY_RESPONSE, 0)
            except Exception:
                pass
        return json.dumps(_EMPTY_RESPONSE, ensure_ascii=False)

    tokenizer, model = _load_model()

    inputs = tokenizer(
        review_text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True,
    )
    if _device is not None and _device != "cpu":
        inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).squeeze().cpu()

    scores_list = probabilities.tolist()
    # Guard: ensure we always have exactly 3 values even if model outputs differ.
    if len(scores_list) != 3:
        scores_list = (scores_list + [0.0, 0.0, 0.0])[:3]

    scores = {label: round(float(scores_list[i]), 6) for i, label in enumerate(_LABEL_ORDER)}

    best_idx = int(probabilities.argmax())
    sentiment = _LABEL_ORDER[best_idx]
    confidence = round(float(probabilities[best_idx]), 6)

    label_used = sentiment if confidence >= _CONFIDENCE_THRESHOLD else "UNCERTAIN"

    return json.dumps(
        {
            "sentiment": sentiment,
            "confidence": confidence,
            "label_used": label_used,
            "scores": scores,
        },
        ensure_ascii=False,
    )
