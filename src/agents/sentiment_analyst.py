"""Factory for the Customer Sentiment Analyst agent.

Wires the review_loader_tool and the analyze_sentiment tool so the agent can
load product reviews from the dataset and classify them via the fine-tuned RoBERTa
3-class sentiment model (DistilBERT is kept as the ablation baseline).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from crewai import Agent

from src.tools import review_loader_tool, sentiment_tool

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "agents.yaml"


def build_sentiment_analyst(llm) -> Agent:
    """Return an Agent instance for sentiment analysis, wired with loader + sentiment tools."""
    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))["sentiment_analyst"]
    return Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        tools=[
            review_loader_tool.load_product_reviews,
            sentiment_tool.analyze_sentiment,
        ],
        allow_delegation=config.get("allow_delegation", False),
        verbose=config.get("verbose", False),
        max_iter=config.get("max_iter", 8),
        llm=llm,
    )
