"""Per-connection GPT-Live session orchestration."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from ..config import Settings
from ..ports.router import RouteDecision
from ..ports.talker import TalkRequest, TranscriptTurn
from .protocol import ClientEvent, ProtocolError, ServerEvent, parse_client_event

if TYPE_CHECKING:
    from ..app import Ports


class WebSocketLike(Protocol):
    """Small WebSocket surface required by the session loop."""

    async def receive_text(self) -> str: ...

    async def send_text(self, data: str) -> None: ...

    async def close(self) -> None: ...


PortFactory = Callable[[], Any]


@dataclass(slots=True)
class SessionState:
    """Mutable state isolated to one live connection."""

    session_id: str = field(default_factory=lambda: f"sess_{uuid4().hex}")
    started: bool = False
    closing: bool = False
    muted: bool = False
    instructions: list[str] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    commentary: list[str] = field(default_factory=list)
    transcripts: list[str] = field(default_factory=list)
    history: list[TranscriptTurn] = field(default_factory=list)
    delegation_ids: set[str] = field(default_factory=set)


class LiveSession:
    """Run one authenticated WebSocket session until it closes."""

    def __init__(self, websocket: WebSocketLike, settings: Settings, port_factory: PortFactory) -> None:
        self.websocket = websocket
        self.settings = settings
        self.port_factory = port_factory
        self.state = SessionState()
        self.ports: Ports | None = None
        self.started_at = time.monotonic()

    async def run(self) -> None:
        """Read, validate, route, and acknowledge events until disconnect."""
        while not self.state.closing:
            try:
                raw = await self.websocket.receive_text()
            except (ConnectionError, EOFError):
                await self.close("connection_lost")
                return

            try:
                event = parse_client_event(raw)
            except ProtocolError as error:
                await self._send(error.as_error())
                continue

            if not self.state.started:
                if event.type != "session.start":
                    await self._send_error("invalid_value", "session.start is required before other events")
                    continue
                await self._start(event)
                continue

            if event.type == "session.start":
                await self._send_error("invalid_value", "session.start may only be sent once")
                continue
            if event.type == "session.close":
                await self.close("close_requested")
                return
            await self._handle(event)

    async def _start(self, event: ClientEvent) -> None:
        """Create this session's ports and announce the resolved session."""
        try:
            self.ports = self.port_factory()
        except Exception:
            await self._send_error("internal_error", "Unable to initialize session adapters")

        self.state.started = True
        data = {
            "session": {
                "id": self.state.session_id,
                "model": event.data["model"],
                "delegation": {"type": "client"},
                "audio": {"format": {"encoding": "pcm16le", "sample_rate": 24000, "channels": 1}},
            }
        }
        if event.event_id is not None:
            data["client_event_id"] = event.event_id
        await self._send(ServerEvent("session.started", data))

    async def _handle(self, event: ClientEvent) -> None:
        """Apply a validated event to this session's isolated state."""
        try:
            if event.type == "session.input_audio.append":
                if not self.state.muted:
                    await self._call_port("append_audio", event.data["audio"])
                return
            if event.type == "session.input_audio.mute":
                self.state.muted = True
                await self._call_port("set_muted", True)
                await self._ack("session.input_audio.muted", event)
                return
            if event.type == "session.input_audio.unmute":
                self.state.muted = False
                await self._call_port("set_muted", False)
                await self._ack("session.input_audio.unmuted", event)
                return
            if event.type in {"session.instructions.append", "session.thinking.append", "session.commentary.append"}:
                text = event.data["text"]
                collection = {
                    "session.instructions.append": self.state.instructions,
                    "session.thinking.append": self.state.thinking,
                    "session.commentary.append": self.state.commentary,
                }[event.type]
                collection.append(text)
                await self._ack(
                    {
                        "session.instructions.append": "session.instructions.appended",
                        "session.thinking.append": "session.thinking.appended",
                        "session.commentary.append": "session.commentary.appended",
                    }[event.type],
                    event,
                )
                return
            if event.type == "session.update":
                await self._ack("session.updated", event)
        except Exception:
            await self._send_error("internal_error", "Session adapter failed")

    async def handle_transcript(self, transcript: str) -> None:
        """Route one completed ASR transcript through the session adapters."""
        if not transcript.strip() or self.ports is None:
            return
        try:
            decision = self.ports.router.classify(transcript)
            if decision.talk:
                reply = await self.ports.talker.reply(
                    TalkRequest(
                        transcript=transcript,
                        instructions=tuple(self.state.instructions),
                        thinking=tuple(self.state.thinking),
                        history=tuple(self.state.history),
                    )
                )
                self.state.history.extend((TranscriptTurn("user", transcript), TranscriptTurn("assistant", reply)))
                self.state.transcripts.extend((transcript, reply))
                await self._send(ServerEvent("session.output_transcript.delta", {"text": reply}))
            else:
                self.state.history.append(TranscriptTurn("user", transcript))
                self.state.transcripts.append(transcript)
            if decision.task:
                delegation_id = await self._create_task(transcript, decision)
                self.state.delegation_ids.add(delegation_id)
                await self._send(
                    ServerEvent(
                        "session.delegation.created",
                        {"delegation_id": delegation_id, "target": "client"},
                    )
                )
        except Exception:
            await self._send_error("internal_error", "Session adapter failed")

    async def _create_task(self, transcript: str, decision: RouteDecision) -> str:
        """Create one outbound task while preserving the router labels."""
        method = getattr(self.ports.tasks, "create", None)
        if method is None:
            raise RuntimeError("task adapter is not configured")
        result = method(self.state.session_id, transcript, dict(decision.labels))
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, str) or not result:
            raise RuntimeError("task adapter returned an invalid delegation id")
        return result

    async def _call_port(self, method_name: str, *args: Any) -> None:
        """Call an optional port method without coupling this layer to adapters."""
        if self.ports is None:
            return
        method = getattr(self.ports.vad, method_name, None)
        if method is None:
            return
        result = method(*args)
        if inspect.isawaitable(result):
            await result

    async def _ack(self, event_type: str, event: ClientEvent) -> None:
        """Send an acknowledgement and preserve the originating client event ID."""
        data = {"client_event_id": event.event_id} if event.event_id is not None else {}
        await self._send(ServerEvent(event_type, data))

    async def _send_error(self, code: str, message: str) -> None:
        """Send a recoverable session error."""
        await self._send(ProtocolError(code, message).as_error())

    async def _send(self, event: ServerEvent) -> None:
        """Serialize and send one server event."""
        await self.websocket.send_text(event.to_json())

    async def close(self, reason: str) -> None:
        """Emit final usage once and close the underlying WebSocket."""
        if self.state.closing:
            return
        self.state.closing = True
        try:
            await self._send(
                ServerEvent(
                    "session.closed",
                    {"usage": {"seconds": int(time.monotonic() - self.started_at)}, "reason": reason},
                )
            )
        except Exception:
            pass
        try:
            await self.websocket.close()
        except Exception:
            pass
