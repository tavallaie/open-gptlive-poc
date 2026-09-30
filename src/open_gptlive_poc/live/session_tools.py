"""Small allow-listed functions owned by one live session."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from loguru import logger


_TOOL_SCHEMAS: tuple[Mapping[str, object], ...] = (
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the current local time and timezone.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_timer",
            "description": "Start a background timer and speak a reminder when it finishes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "integer", "minimum": 1, "maximum": 300},
                    "message": {"type": "string", "description": "What to remind the user about."},
                },
                "required": ["seconds", "message"],
                "additionalProperties": False,
            },
        },
    },
)


class SessionTools:
    """Expose clock and background timer functions for one voice session."""

    def __init__(
        self,
        notify: Callable[[str], bool],
        *,
        on_timer_started: Callable[[int, str], None] | None = None,
        session_id: str | None = None,
    ) -> None:
        self._notify = notify
        self._on_timer_started = on_timer_started
        self.log = logger.bind(component="session-tools", session_id=session_id)
        self._timers: set[asyncio.Task[None]] = set()

    @property
    def schemas(self) -> Sequence[Mapping[str, object]]:
        return _TOOL_SCHEMAS

    async def execute(self, name: str, arguments: Mapping[str, object]) -> str:
        started = time.perf_counter()
        self.log.info("Tool execution started", tool_name=name)
        try:
            if name == "get_current_time":
                result = datetime.now().astimezone().isoformat(timespec="seconds")
            elif name == "start_timer":
                result = self._start_timer(arguments)
            else:
                raise ValueError(f"Unknown function: {name}")
        except Exception:
            self.log.exception("Tool execution failed", tool_name=name)
            raise
        self.log.info(
            "Tool execution completed",
            tool_name=name,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )
        return result

    async def close(self) -> None:
        """Cancel timers when their live session ends."""
        for task in self._timers:
            task.cancel()
        await asyncio.gather(*self._timers, return_exceptions=True)
        self._timers.clear()

    def _start_timer(self, arguments: Mapping[str, object]) -> str:
        seconds = arguments.get("seconds")
        message = arguments.get("message")
        if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= 300:
            raise ValueError("seconds must be an integer from 1 to 300")
        if not isinstance(message, str) or not message.strip() or len(message) > 120:
            raise ValueError("message must be a non-empty string of at most 120 characters")
        task = asyncio.create_task(self._finish_timer(seconds, message.strip()))
        self._timers.add(task)
        task.add_done_callback(self._timers.discard)
        if self._on_timer_started is not None:
            self._on_timer_started(seconds, message.strip())
        return f"Timer started for {seconds} seconds. I will remind the user: {message.strip()}"

    async def _finish_timer(self, seconds: int, message: str) -> None:
        self.log.info("Background timer started", seconds=seconds)
        await asyncio.sleep(seconds)
        accepted = self._notify(f"Timer finished: {message}")
        self.log.info("Background timer completed", seconds=seconds, notification_queued=accepted)
