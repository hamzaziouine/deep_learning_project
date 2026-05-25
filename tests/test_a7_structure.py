"""Structure tests for A7 — agent definitions, YAML configs, and main.py entry point.

These tests do NOT call Gemini or run the crew end-to-end; they verify the wiring
is in place so a live run becomes possible once the Gemini key is rotated.
"""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_agents_yaml_defines_three_roles():
    config = yaml.safe_load((REPO_ROOT / "config" / "agents.yaml").read_text(encoding="utf-8"))
    assert set(config.keys()) >= {"market_researcher", "sentiment_analyst", "manager"}
    for name in ("market_researcher", "sentiment_analyst", "manager"):
        spec = config[name]
        assert "role" in spec and spec["role"]
        assert "goal" in spec and spec["goal"]
        assert "backstory" in spec and spec["backstory"]


def test_tasks_yaml_defines_three_tasks_in_order():
    config = yaml.safe_load((REPO_ROOT / "config" / "tasks.yaml").read_text(encoding="utf-8"))
    assert "market_research_task" in config
    assert "sentiment_analysis_task" in config
    assert "synthesis_task" in config
    # Synthesis task must trigger HITL pause.
    assert config["synthesis_task"].get("human_input") is True
    # Each task must declare an expected_output for the LLM.
    for task in config.values():
        assert "description" in task
        assert "expected_output" in task


def test_agent_factories_importable():
    """Importing the factories must not require Gemini auth — only YAML + crewai class."""
    from src.agents.market_researcher import build_market_researcher  # noqa: F401
    from src.agents.sentiment_analyst import build_sentiment_analyst  # noqa: F401
    from src.agents.manager import build_manager  # noqa: F401


def test_main_module_exposes_full_pipeline_entrypoints():
    """main.py must expose both the smoke crew and the full pipeline."""
    from src import main as main_mod

    assert hasattr(main_mod, "run_smoke_crew")
    assert hasattr(main_mod, "run_market_intelligence_crew")
    assert hasattr(main_mod, "main")


def test_market_research_task_references_niche_placeholder():
    config = yaml.safe_load((REPO_ROOT / "config" / "tasks.yaml").read_text(encoding="utf-8"))
    desc = config["market_research_task"]["description"]
    assert "{niche}" in desc, "market_research_task must allow niche interpolation"


def test_sentiment_task_has_market_context_dependency():
    """Sentiment task should reference the market task as context (richer signal for the analyst)."""
    config = yaml.safe_load((REPO_ROOT / "config" / "tasks.yaml").read_text(encoding="utf-8"))
    deps = config["sentiment_analysis_task"].get("context_tasks", [])
    assert "market_research_task" in deps


def test_set_logger_callable_on_each_tool():
    """Tool modules must expose set_logger() so A7 can wire AgentLogger at crew setup."""
    from src.tools import sentiment_tool, web_search_tool, review_loader_tool

    for module in (sentiment_tool, web_search_tool, review_loader_tool):
        assert callable(getattr(module, "set_logger", None)), (
            f"{module.__name__} missing set_logger"
        )
