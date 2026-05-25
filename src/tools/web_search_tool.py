"""
Web search tool — wraps DuckDuckGo search for CrewAI market research agent.

Rate policy: 1 request per 2 seconds (self-imposed floor).
Backoff policy: 2s, 4s, 8s on RatelimitException, max 3 retries.
On persistent failure the tool returns a controlled error dict; no exception escapes.
"""

import json
import logging
import time

from crewai.tools import tool
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException

logger = logging.getLogger(__name__)

_MIN_INTERVAL_S = 2.0          # self-imposed floor between requests
_BACKOFF_DELAYS = (2, 4, 8)    # exponential delays on RatelimitException
_MAX_RESULTS = 10

# A7 wires the AgentLogger here at crew setup; tool works without it.
_agent_logger = None


def set_logger(agent_logger) -> None:
    """A7 hook: register an AgentLogger for tool-call traceability."""
    global _agent_logger
    _agent_logger = agent_logger


@tool("search_market")
def search_market(query: str) -> dict:
    """Search the web for market and competitor information. Returns search result snippets only."""
    time.sleep(_MIN_INTERVAL_S)

    for attempt, backoff in enumerate(_BACKOFF_DELAYS, start=1):
        try:
            raw = DDGS().text(query, max_results=_MAX_RESULTS)
        except RatelimitException as exc:
            logger.warning(
                "RatelimitException on attempt %d for query %r: %s — backing off %ds",
                attempt, query, exc, backoff,
            )
            time.sleep(backoff)
            continue
        except Exception as exc:
            logger.error("Unexpected error searching %r: %s", query, exc, exc_info=True)
            return json.dumps(
                {
                    "results": [],
                    "query": query,
                    "result_count": 0,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )

        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "snippet": item.get("body", ""),
            }
            for item in (raw or [])
        ]
        return json.dumps(
            {
                "results": results,
                "query": query,
                "result_count": len(results),
            },
            ensure_ascii=False,
        )

    # All retries exhausted
    msg = f"DuckDuckGo rate limit persisted after {len(_BACKOFF_DELAYS)} retries for query: {query!r}"
    logger.error(msg)
    return json.dumps(
        {
            "results": [],
            "query": query,
            "result_count": 0,
            "error": msg,
        },
        ensure_ascii=False,
    )
