"""Small allow-listed functions owned by one live session."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from loguru import logger

from ..ports.router import ToolProfile


_TOOL_SCHEMAS: tuple[Mapping[str, object], ...] = (
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the current local time and timezone. Call this when the user asks what time it is.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_timer",
            "description": "Start one background timer, alarm, or reminder for this user request and deliver its result when it finishes. Call this instead of merely describing or repeating the requested timer.",
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

_TOOL_PROFILES = {
    "get_current_time": ToolProfile("inline", "urgent"),
    "start_timer": ToolProfile("background", "urgent", can_request_input=False),
}


class SessionTools:
    """Expose clock and background timer functions for one voice session."""

    def __init__(
        self,
        notify: Callable[[str, str, ToolProfile], bool],
        *,
        on_timer_started: Callable[[int, str], None] | None = None,
        session_id: str | None = None,
    ) -> None:
        self._notify = notify
        self._on_timer_started = on_timer_started
        self._request_context = ""
        self._request_generation = 0
        self._timer_scheduled_for_request = False
        self._urgent_tool_called = False
        self.log = logger.bind(component="session-tools", session_id=session_id)
        self._timers: set[asyncio.Task[None]] = set()
        self._timer_jobs: dict[tuple[int, int, str], asyncio.Task[None]] = {}

    @property
    def schemas(self) -> Sequence[Mapping[str, object]]:
        return _TOOL_SCHEMAS

    @property
    def profiles(self) -> Mapping[str, ToolProfile]:
        """Profiles supplied to routing and background-result classification."""
        return _TOOL_PROFILES

    @property
    def urgent_tool_called(self) -> bool:
        """Whether this request actually invoked a tool marked urgent."""
        return self._urgent_tool_called

    def set_request_context(self, context: str) -> None:
        """Remember the request context for any background work started now."""
        self._request_generation += 1
        self._timer_scheduled_for_request = False
        self._urgent_tool_called = False
        self._request_context = context

    def profile(self, name: str) -> ToolProfile:
        """Return runtime metadata for a known tool."""
        try:
            return _TOOL_PROFILES[name]
        except KeyError as exc:
            raise ValueError(f"Unknown function: {name}") from exc

    async def execute(self, name: str, arguments: Mapping[str, object]) -> str:
        started = time.perf_counter()
        self.log.info("Tool execution started", tool_name=name)
        try:
            profile = self.profile(name)
            self._urgent_tool_called = self._urgent_tool_called or profile.urgency == "urgent"
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
        message = message.strip()
        if self._timer_scheduled_for_request:
            self.log.warning("Additional timer call ignored for the same user request", seconds=seconds)
            return "A timer has already been scheduled for this user request. Do not start another timer."
        key = (self._request_generation, seconds, message)
        existing = self._timer_jobs.get(key)
        if existing is not None and not existing.done():
            self.log.warning("Duplicate timer call ignored", seconds=seconds)
            return f"That {seconds}-second timer is already running."
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds
        task = asyncio.create_task(
            self._finish_timer(seconds, message, self._request_context, key, deadline)
        )
        self._timers.add(task)
        self._timer_jobs[key] = task
        self._timer_scheduled_for_request = True
        task.add_done_callback(self._timers.discard)
        self.log.info("Background timer scheduled", seconds=seconds)
        if self._on_timer_started is not None:
            self._on_timer_started(seconds, message.strip())
        return f"Timer started for {seconds} seconds. I will remind the user: {message.strip()}"

    async def _finish_timer(
        self,
        seconds: int,
        message: str,
        request: str,
        key: tuple[int, int, str],
        deadline: float,
    ) -> None:
        loop = asyncio.get_running_loop()
        started_at = deadline - seconds
        try:
            await asyncio.sleep(max(0, deadline - loop.time()))
            accepted = self._notify(
                f"Timer finished: {message}", request, self.profile("start_timer")
            )
            self.log.info(
                "Background timer completed",
                seconds=seconds,
                elapsed_ms=round((loop.time() - started_at) * 1000),
                notification_queued=accepted,
            )
        finally:
            if self._timer_jobs.get(key) is asyncio.current_task():
                self._timer_jobs.pop(key, None)
