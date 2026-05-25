"""JSONL agent logger. Thread-safe. One file per run."""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from config.settings import LOGS_DIR


class AgentLogger:
    def __init__(self, run_id: str = None):
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = LOGS_DIR
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"{self.run_id}.jsonl"
        self._lock = threading.Lock()

    def log(
        self,
        agent: str,
        action: str,
        input_data: dict,
        output_data: dict = None,
        status: str = "success",
        error: str = None,
        duration_ms: int = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
            "input": input_data,
            "output": output_data,
            "duration_ms": duration_ms,
            "status": status,
            "error": error,
        }
        with self._lock:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- 8 convenience methods (one per spec §8.2 event type) ---

    def crew_start(self, niche: str, config: dict) -> None:
        self.log(
            agent="manager",
            action="crew_start",
            input_data={"niche": niche, "config": config},
        )

    def task_delegated(self, task_description: str, assigned_agent: str) -> None:
        self.log(
            agent="manager",
            action="task_delegated",
            input_data={
                "task_description": task_description,
                "assigned_agent": assigned_agent,
            },
        )

    def tool_called(
        self,
        tool_name: str,
        input_data: dict,
        output_data: dict,
        duration_ms: int,
        agent: str = "unknown",
    ) -> None:
        self.log(
            agent=agent,
            action="tool_called",
            input_data={"tool_name": tool_name, "input": input_data},
            output_data=output_data,
            duration_ms=duration_ms,
        )

    def tool_error(
        self,
        tool_name: str,
        input_data: dict,
        error_message: str,
        traceback_str: str = None,
        agent: str = "unknown",
    ) -> None:
        self.log(
            agent=agent,
            action="tool_error",
            input_data={"tool_name": tool_name, "input": input_data},
            output_data={"traceback": traceback_str},
            status="error",
            error=error_message,
        )

    def hitl_checkpoint(self, summary: str, user_feedback: str) -> None:
        self.log(
            agent="manager",
            action="hitl_checkpoint",
            input_data={"summary": summary},
            output_data={"user_feedback": user_feedback},
        )

    def agent_response(
        self, agent_name: str, response_text: str
    ) -> None:
        self.log(
            agent=agent_name,
            action="agent_response",
            input_data={},
            output_data={"response_text": response_text},
        )

    def report_generated(self, report_path: str, word_count: int) -> None:
        self.log(
            agent="manager",
            action="report_generated",
            input_data={},
            output_data={"report_path": report_path, "word_count": word_count},
        )

    def crew_complete(self, total_duration_ms: int, status: str = "success") -> None:
        self.log(
            agent="manager",
            action="crew_complete",
            input_data={},
            output_data={"total_duration_ms": total_duration_ms},
            status=status,
            duration_ms=total_duration_ms,
        )
