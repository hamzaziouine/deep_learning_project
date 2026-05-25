"""Tests for AgentLogger (src/utils/logger.py)."""
import json
import concurrent.futures
from pathlib import Path

import pytest

from src.utils.logger import AgentLogger


# ---------------------------------------------------------------------------
# Test 1 — file creation + basic JSONL entry
# ---------------------------------------------------------------------------

def test_creates_jsonl_file_and_readable_entry(tmp_path, monkeypatch):
    """Logger creates a .jsonl file; each line is valid JSON with required fields."""
    # Redirect LOGS_DIR so we don't pollute the real logs/ folder during tests.
    import src.utils.logger as logger_module
    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)

    run_id = "run_test_basic"
    logger = AgentLogger(run_id=run_id)

    logger.log(
        agent="test_agent",
        action="test_action",
        input_data={"key": "value"},
        output_data={"result": 42},
        status="success",
        duration_ms=10,
    )

    log_file = tmp_path / f"{run_id}.jsonl"
    assert log_file.exists(), "JSONL file was not created"

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, "Expected exactly 1 log line"

    entry = json.loads(lines[0])
    required = {"timestamp", "run_id", "agent", "action", "input", "output",
                "duration_ms", "status", "error"}
    assert required.issubset(entry.keys()), f"Missing fields: {required - entry.keys()}"

    assert entry["run_id"] == run_id
    assert entry["agent"] == "test_agent"
    assert entry["action"] == "test_action"
    assert entry["status"] == "success"
    assert entry["output"] == {"result": 42}


# ---------------------------------------------------------------------------
# Test 2 — all 8 convenience methods produce the correct action string
# ---------------------------------------------------------------------------

def test_all_eight_event_types_produce_correct_action(tmp_path, monkeypatch):
    """Each convenience method writes an entry whose action field matches spec §8.2."""
    import src.utils.logger as logger_module
    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)

    logger = AgentLogger(run_id="run_test_events")

    logger.crew_start(niche="wireless earbuds", config={"agents": 2})
    logger.task_delegated(task_description="Research competitors", assigned_agent="market_researcher")
    logger.tool_called(tool_name="search_market", input_data={"query": "earbuds"}, output_data={"results": []}, duration_ms=150)
    logger.tool_error(tool_name="search_market", input_data={"query": "earbuds"}, error_message="timeout", traceback_str="Traceback...")
    logger.hitl_checkpoint(summary="Research complete", user_feedback="approve")
    logger.agent_response(agent_name="sentiment_analyst", response_text="70% positive")
    logger.report_generated(report_path="outputs/report.md", word_count=850)
    logger.crew_complete(total_duration_ms=45000, status="success")

    log_file = tmp_path / "run_test_events.jsonl"
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8, f"Expected 8 entries, got {len(lines)}"

    actions = [json.loads(line)["action"] for line in lines]
    expected_actions = [
        "crew_start",
        "task_delegated",
        "tool_called",
        "tool_error",
        "hitl_checkpoint",
        "agent_response",
        "report_generated",
        "crew_complete",
    ]
    assert actions == expected_actions, f"Action mismatch: {actions}"

    # tool_error must have status="error"
    tool_error_entry = json.loads(lines[3])
    assert tool_error_entry["status"] == "error"
    assert tool_error_entry["error"] == "timeout"


# ---------------------------------------------------------------------------
# Test 3 — thread safety: 50 concurrent log() calls, no corruption
# ---------------------------------------------------------------------------

def test_concurrent_writes_are_all_present_and_valid(tmp_path, monkeypatch):
    """50 threads logging simultaneously — all entries land in file, none corrupted."""
    import src.utils.logger as logger_module
    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)

    logger = AgentLogger(run_id="run_test_concurrent")
    call_count = 50

    def write_entry(index: int):
        logger.log(
            agent=f"agent_{index}",
            action="tool_called",
            input_data={"index": index},
            output_data={"done": True},
            duration_ms=index,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(write_entry, i) for i in range(call_count)]
        concurrent.futures.wait(futures)

    log_file = tmp_path / "run_test_concurrent.jsonl"
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == call_count, f"Expected {call_count} entries, got {len(lines)}"

    # Every line must be valid JSON with the required fields.
    for line in lines:
        entry = json.loads(line)  # raises if corrupted
        assert "run_id" in entry
        assert "timestamp" in entry

    # All indices 0-49 must be present (no dropped writes).
    indices_written = {json.loads(line)["input"]["index"] for line in lines}
    assert indices_written == set(range(call_count)), "Some entries were lost"


# ---------------------------------------------------------------------------
# Test 4 — custom run_id is honoured
# ---------------------------------------------------------------------------

def test_custom_run_id_is_honoured(tmp_path, monkeypatch):
    """When a run_id is supplied, it appears in the filename and in every entry."""
    import src.utils.logger as logger_module
    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)

    custom_id = "run_my_custom_id_2026"
    logger = AgentLogger(run_id=custom_id)
    logger.log(agent="mgr", action="crew_start", input_data={"niche": "test"})

    expected_file = tmp_path / f"{custom_id}.jsonl"
    assert expected_file.exists(), "File not named after custom run_id"

    entry = json.loads(expected_file.read_text(encoding="utf-8").strip())
    assert entry["run_id"] == custom_id


# ---------------------------------------------------------------------------
# Test 5 — auto run_id contains "run_" prefix when not supplied
# ---------------------------------------------------------------------------

def test_auto_run_id_has_run_prefix(tmp_path, monkeypatch):
    """Auto-generated run_id starts with 'run_' per spec §8.3."""
    import src.utils.logger as logger_module
    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)

    logger = AgentLogger()
    assert logger.run_id.startswith("run_"), f"run_id does not start with 'run_': {logger.run_id}"
