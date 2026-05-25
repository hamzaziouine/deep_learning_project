"""Multi-Agent Product Review Intelligence — main entry point.

Usage:
    python -m src.main --niche "wireless earbuds" [--category Electronics]
    python -m src.main --niche "wireless earbuds" --no-hitl   # skip HITL pause
    python -m src.main --smoke                                # smoke test
    python -m src.main --validate-key                         # Gemini key check
    python -m src.main --replay demo/wireless_earbuds_demo    # replay saved run

The full pipeline is hierarchical: a manager agent delegates to a Market Research
Analyst (uses search_market) and a Customer Sentiment Analyst (uses
load_product_reviews + analyze_sentiment). The synthesis step pauses for
human-in-the-loop approval before the final report is written to outputs/.

--replay PATH_PREFIX loads PATH_PREFIX.md (the report) and PATH_PREFIX.jsonl
(the run log) and prints them to stdout WITHOUT making any LLM calls. This is
the defense-time fallback when Gemini quota is exhausted or network is down.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Defense-day quiet mode: suppress noisy library warnings, HF/torch progress
# bars, and the google-genai dual-key logger warning. Crewai re-calls
# load_dotenv() on import which re-injects GEMINI_API_KEY, so popping it is
# useless — silencing the logger is the only durable fix.
# Override with QUIET_DEMO=0 to restore full verbosity.
if os.environ.get("QUIET_DEMO", "1") != "0":
    import logging
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.getLogger("google.genai").setLevel(logging.ERROR)
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)

from dotenv import load_dotenv

load_dotenv()

# google-genai SDK reads both keys and warns when they differ; keep them in sync.
gemini_key = os.environ.get("GEMINI_API_KEY", "")
if gemini_key:
    os.environ["GOOGLE_API_KEY"] = gemini_key

# litellm (CrewAI's backend) supports disk caching out of the box. With the free-tier
# quota of 20 req/day, identical prompts hitting the cache instead of the API is a
# meaningful win during dev / demo / replay scenarios.
os.environ.setdefault("LITELLM_CACHE", "disk")

# noqa: E402 — imports below must come after env setup
from crewai import Agent, Crew, LLM, Process, Task  # noqa: E402
import yaml  # noqa: E402

from config.settings import DEFAULT_GEMINI_MODEL, OUTPUTS_DIR, LOGS_DIR  # noqa: E402
from src.utils.logger import AgentLogger  # noqa: E402
from src.tools import sentiment_tool, web_search_tool, review_loader_tool  # noqa: E402
from src.agents.market_researcher import build_market_researcher  # noqa: E402
from src.agents.sentiment_analyst import build_sentiment_analyst  # noqa: E402
from src.agents.manager import build_manager  # noqa: E402

_TASKS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "tasks.yaml"


# ---------------------------------------------------------------------------
# Rate-limit guard — Gemini free-tier per-minute caps (5 RPM on some variants)
# burn under CrewAI's automatic 429 retry. We enforce a floor between calls
# so the crew never bursts faster than the cap.
# ---------------------------------------------------------------------------

_LLM_MIN_INTERVAL_S = float(os.environ.get("LLM_MIN_INTERVAL_S", "0"))
_LAST_LLM_CALL_TS = 0.0


def _install_llm_throttle(llm) -> None:
    """Wrap llm.call so each invocation waits LLM_MIN_INTERVAL_S since the last.

    No-op when LLM_MIN_INTERVAL_S = 0 (default). Set to e.g. 13 to stay below
    5 RPM for gemini-flash-latest (gemini-3-flash) free tier.
    """
    if _LLM_MIN_INTERVAL_S <= 0:
        return

    original_call = llm.call

    def throttled_call(*args, **kwargs):
        global _LAST_LLM_CALL_TS
        now = time.time()
        elapsed = now - _LAST_LLM_CALL_TS
        wait_s = _LLM_MIN_INTERVAL_S - elapsed
        if wait_s > 0:
            time.sleep(wait_s)
        _LAST_LLM_CALL_TS = time.time()
        return original_call(*args, **kwargs)

    llm.call = throttled_call


# ---------------------------------------------------------------------------
# Smoke crew (preserved so test_crew_smoke.py keeps working once the key is rotated)
# ---------------------------------------------------------------------------


def run_smoke_crew() -> object:
    """Minimal hierarchical crew to verify the stack reaches Gemini.

    Returns the CrewOutput object (str(result) gives the text).
    """
    llm = LLM(model=DEFAULT_GEMINI_MODEL, api_key=gemini_key)

    greeter = Agent(
        role="Greeter",
        goal="Respond with a short greeting to confirm the system is operational.",
        backstory="You are a minimal verification agent that proves the CrewAI stack calls Gemini correctly.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    task = Task(
        description="Reply with a single short greeting sentence (e.g. 'Hello, system is operational.').",
        expected_output="A non-empty greeting string.",
        agent=greeter,
    )

    crew = Crew(
        agents=[greeter],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=False,
    )

    return crew.kickoff()


# ---------------------------------------------------------------------------
# Full multi-agent pipeline
# ---------------------------------------------------------------------------


def _load_task_config() -> dict:
    return yaml.safe_load(_TASKS_CONFIG_PATH.read_text(encoding="utf-8"))


def _build_tasks(
    task_config: dict, agents: dict, niche: str, category: str, hitl: bool = True
) -> list[Task]:
    """Materialize Task objects from YAML, in the order: market -> sentiment -> synthesis.

    When hitl=False, synthesis_task runs end-to-end without pausing for approval.
    """
    market = Task(
        description=task_config["market_research_task"]["description"].format(niche=niche),
        expected_output=task_config["market_research_task"]["expected_output"],
        agent=agents["market_researcher"],
    )
    sentiment = Task(
        description=task_config["sentiment_analysis_task"]["description"].format(
            niche=niche, category=category
        ),
        expected_output=task_config["sentiment_analysis_task"]["expected_output"],
        agent=agents["sentiment_analyst"],
        context=[market],
    )
    human_input_flag = (
        bool(task_config["synthesis_task"].get("human_input", True)) if hitl else False
    )
    synthesis = Task(
        description=task_config["synthesis_task"]["description"].format(niche=niche),
        expected_output=task_config["synthesis_task"]["expected_output"],
        agent=agents["manager"],
        context=[market, sentiment],
        human_input=human_input_flag,
    )
    return [market, sentiment, synthesis]


# ---------------------------------------------------------------------------
# HITL approval helper — accept short text strings as approval, not feedback.
# ---------------------------------------------------------------------------

# Tokens d'approbation acceptés au prompt HITL (insensible à la casse, trim
# auto). Inclut variantes anglaises et françaises pour robustesse en démo de
# défense. Tout autre texte non vide est traité par CrewAI comme un feedback
# et relance le manager.
#
# Liste exhaustive (ne pas étendre sans relire la logique d'override de
# `builtins.input` dans `_install_hitl_approval_filter`) :
#   anglais : "" / "approve" / "approved" / "ok" / "okay" / "yes" / "y"
#   français : "valider" / "ok valider" / "valide" / "validé"
#
# CrewAI by default treats ANY non-empty input as feedback and re-runs the
# manager. That kicks the crew into a max_iter loop if the examiner types
# "approve" or "ok". Normalize these to empty so CrewAI accepts and continues.
_APPROVAL_TOKENS = {
    "",
    "approve",
    "approved",
    "ok",
    "okay",
    "yes",
    "y",
    "valider",
    "ok valider",
    "valide",
    "validé",
}


def _install_hitl_approval_filter() -> None:
    """Wrap builtins.input so approval-style strings are converted to empty input.

    CrewAI's HITL prompt reads via builtins.input. By intercepting and replacing
    approval tokens with "", we make the prompt accept "approve"/"ok"/"y"/etc.
    as a clean approval and let the crew finalize the report.

    Anything not in _APPROVAL_TOKENS is passed through untouched (= real feedback).
    """
    import builtins

    if getattr(builtins, "_dlp_input_patched", False):
        return
    original_input = builtins.input

    def patched_input(prompt: str = "") -> str:
        raw = original_input(prompt)
        normalized = (raw or "").strip().lower()
        if normalized in _APPROVAL_TOKENS:
            return ""
        return raw

    builtins.input = patched_input  # type: ignore[assignment]
    builtins._dlp_input_patched = True  # type: ignore[attr-defined]


def run_market_intelligence_crew(
    niche: str,
    category: str = "Electronics",
    verbose: bool = False,
    hitl: bool = True,
) -> str:
    """Run the full hierarchical pipeline and write the report to outputs/.

    Returns the path of the saved report.
    """
    if not gemini_key:
        raise RuntimeError(
        )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"agent_{timestamp}"

    # AgentLogger appends `.jsonl` itself — pass just the run_id, not the path.
    logger = AgentLogger(run_id)
    logger.crew_start(niche=niche, config={"category": category, "model": DEFAULT_GEMINI_MODEL})

    sentiment_tool.set_logger(logger)
    web_search_tool.set_logger(logger)
    review_loader_tool.set_logger(logger)

    llm = LLM(model=DEFAULT_GEMINI_MODEL, api_key=gemini_key)
    _install_llm_throttle(llm)  # no-op unless LLM_MIN_INTERVAL_S env var > 0

    agents = {
        "market_researcher": build_market_researcher(llm),
        "sentiment_analyst": build_sentiment_analyst(llm),
        "manager": build_manager(llm),
    }

    task_config = _load_task_config()
    tasks = _build_tasks(task_config, agents, niche=niche, category=category, hitl=hitl)

    if hitl:
        _install_hitl_approval_filter()

    crew = Crew(
        agents=[agents["market_researcher"], agents["sentiment_analyst"]],
        tasks=tasks,
        process=Process.hierarchical,
        manager_agent=agents["manager"],
        verbose=verbose,
    )

    started = time.time()
    try:
        result = crew.kickoff()
    except Exception as exc:
        elapsed_ms = int((time.time() - started) * 1000)
        logger.crew_complete(total_duration_ms=elapsed_ms, status="error")
        raise RuntimeError(
            f"Crew execution failed after {elapsed_ms} ms: {exc}\n"
            "Fallback for live demo: "
            "python -m src.main --replay demo/wireless_earbuds_demo"
        ) from exc
    elapsed_ms = int((time.time() - started) * 1000)

    report_md = str(result)
    report_path = OUTPUTS_DIR / f"{niche.replace(' ', '_')}_{timestamp}.md"
    report_path.write_text(report_md, encoding="utf-8")

    logger.report_generated(report_path=str(report_path), word_count=len(report_md.split()))
    logger.crew_complete(total_duration_ms=elapsed_ms, status="success")

    return str(report_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Replay mode — defense-time fallback. NO LLM calls.
# ---------------------------------------------------------------------------


def run_replay(path_prefix: str) -> int:
    """Print a saved demo run (report .md + run log .jsonl) without LLM calls.

    path_prefix is the path stem; we read {prefix}.md and (optionally) {prefix}.jsonl.
    Returns 0 on success, 1 if the .md is missing.
    """
    prefix = Path(path_prefix)
    md_path = prefix.with_suffix(".md")
    jsonl_path = prefix.with_suffix(".jsonl")

    if not md_path.is_file():
        print(
            f"ERROR: replay file not found: {md_path}\n"
            f"Expected a sibling .md (and optional .jsonl) at the given path prefix.",
            file=sys.stderr,
        )
        return 1

    print("=" * 72)
    print("REPLAY MODE — no LLM calls made")
    print(f"Source report: {md_path}")
    if jsonl_path.is_file():
        print(f"Source log:    {jsonl_path}")
    print("=" * 72)
    print()

    print(md_path.read_text(encoding="utf-8"))

    if jsonl_path.is_file():
        print()
        print("-" * 72)
        print("Run log (JSONL):")
        print("-" * 72)
        print(jsonl_path.read_text(encoding="utf-8"))
    else:
        print()
        print(f"(no .jsonl log found at {jsonl_path}; skipping run-log replay)")

    return 0


def _validate_gemini_key() -> bool:
    """Fail-fast preflight: a single lightweight Gemini call to confirm the key is alive.

    Returns True if the key works, False otherwise. Used by --validate-key.
    """
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY missing in .env", file=sys.stderr)
        return False
    try:
        # Use a 1-token completion as the cheapest probe.
        from google import genai  # type: ignore

        client = genai.Client(api_key=gemini_key)
        # models.list() is a free metadata call — 0 generation cost.
        list(client.models.list())
        print("Gemini key OK.")
        return True
    except Exception as exc:  # pragma: no cover — diagnostic path
        print(f"Gemini key validation failed: {exc}", file=sys.stderr)
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Product Review Intelligence pipeline."
    )
    parser.add_argument(
        "--niche", type=str, help="Product niche to analyze (e.g. 'wireless earbuds')."
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Electronics",
        help="Dataset category to filter reviews on (default: Electronics).",
    )
    parser.add_argument("--smoke", action="store_true", help="Run smoke crew instead of full pipeline.")
    parser.add_argument(
        "--validate-key",
        action="store_true",
        help="Validate the Gemini API key with a single lightweight call, then exit.",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        metavar="PATH_PREFIX",
        help=(
            "Replay a saved demo run from PATH_PREFIX.md (+ optional PATH_PREFIX.jsonl). "
            "No LLM calls are made. Defense-time fallback."
        ),
    )
    parser.add_argument(
        "--no-hitl",
        action="store_true",
        help="Disable the HITL pause on the synthesis task (run end-to-end).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.replay:
        return run_replay(args.replay)

    if args.validate_key:
        return 0 if _validate_gemini_key() else 1

    if args.smoke:
        result = run_smoke_crew()
        print(result)
        return 0

    if not args.niche:
        print(
            "ERROR: --niche is required (or use --smoke / --validate-key / --replay).",
            file=sys.stderr,
        )
        return 2

    report_path = run_market_intelligence_crew(
        niche=args.niche,
        category=args.category,
        verbose=args.verbose,
        hitl=not args.no_hitl,
    )
    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
