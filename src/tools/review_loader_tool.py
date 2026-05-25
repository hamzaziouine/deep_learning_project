"""Review loader tool — loads product reviews from a static CSV dataset."""
import json
import os

import pandas as pd
from crewai.tools import tool

from config import settings

# Resolved at call time so that DATA_PATH env var changes (e.g. in tests)
# take effect even after crewai has already called load_dotenv() at import.
_DEFAULT_CSV = str(settings.DATA_SAMPLE / "electronics_100.csv")

# A7 wires the AgentLogger here at crew setup; tool works without it.
_agent_logger = None


def set_logger(agent_logger) -> None:
    """A7 hook: register an AgentLogger for tool-call traceability."""
    global _agent_logger
    _agent_logger = agent_logger


def _resolve_data_path() -> str:
    """Return a readable CSV path.

    Prefers the DATA_PATH env var when it points to an existing file.
    Falls back to the bundled sample CSV when DATA_PATH is absent,
    points to a directory (the .env.example default), or does not exist.
    """
    from pathlib import Path

    candidate = os.environ.get("DATA_PATH", "")
    if candidate and Path(candidate).is_file():
        return candidate
    return _DEFAULT_CSV


@tool("load_product_reviews")
def load_product_reviews(category: str, max_reviews: int = 50) -> dict:
    """Load product reviews from a static dataset for the given category."""
    data_path = _resolve_data_path()
    try:
        df = pd.read_csv(data_path)
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return json.dumps(
            {
                "reviews": [],
                "total_available": 0,
                "category": category,
                "error": f"Dataset not found at {data_path}",
            },
            ensure_ascii=False,
        )

    available_categories = sorted(df["category"].dropna().unique().tolist())

    if category not in available_categories:
        return json.dumps(
            {
                "reviews": [],
                "total_available": 0,
                "category": category,
                "error": f"Unknown category. Available: {available_categories}",
            },
            ensure_ascii=False,
        )

    category_df = df[df["category"] == category]
    total_available = len(category_df)

    # Hard cap at 10 reviews to protect Gemini free-tier quota — see council
    # day 4 architect verdict; agent could otherwise call analyze_sentiment 50x.
    capped_max = min(int(max_reviews), 10)
    selected = category_df.head(capped_max)

    reviews = [
        {
            "text": str(row["text"]),
            "rating": int(row["rating"]),
            "helpful_votes": int(row["helpful_votes"]),
            "verified": bool(row["verified"]),
        }
        for _, row in selected.iterrows()
    ]

    return json.dumps(
        {
            "reviews": reviews,
            "total_available": total_available,
            "category": category,
        },
        ensure_ascii=False,
    )
