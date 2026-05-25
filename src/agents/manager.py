"""Factory for the Product Intelligence Coordinator (manager agent).

Used with `manager_agent=` (not `manager_llm=`) on the Crew so the explicit
delegation-oriented backstory is enforced. Per orchestration architect:
auto-managers tend to answer questions directly when they have enough context;
the explicit "NEVER answer yourself" instruction is what makes hierarchical
delegation actually happen.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from crewai import Agent

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "agents.yaml"


def build_manager(llm) -> Agent:
    """Return an Agent instance for hierarchical orchestration. No tools — pure delegation."""
    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))["manager"]
    return Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        tools=[],
        allow_delegation=config.get("allow_delegation", True),
        verbose=config.get("verbose", False),
        max_iter=config.get("max_iter", 6),
        llm=llm,
    )
