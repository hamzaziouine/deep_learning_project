"""Factory for the Market Research Analyst agent.

Loads role/goal/backstory from config/agents.yaml so the YAML stays the single
source of truth (matching the spec's requirement to externalize agent config).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from crewai import Agent

from src.tools import web_search_tool

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "agents.yaml"


def build_market_researcher(llm) -> Agent:
    """Return an Agent instance for market research, wired with the search_market tool."""
    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))["market_researcher"]
    return Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        tools=[web_search_tool.search_market],
        allow_delegation=config.get("allow_delegation", False),
        verbose=config.get("verbose", False),
        max_iter=config.get("max_iter", 8),
        llm=llm,
    )
