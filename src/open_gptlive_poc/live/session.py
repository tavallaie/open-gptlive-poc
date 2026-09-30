"""Per-connection GPT-Live session orchestration."""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from threading import Event as ThreadEvent
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from loguru import logger

from ..config import Settings
from ..ports.router import RouteDecision, ToolProfile
from ..ports.talker import TalkRequest, TranscriptTurn
from .protocol import ClientEvent, ProtocolError, ServerEvent, parse_client_event
from .session_tools import SessionTools

if TYPE_CHECKING:
    from ..app import Ports


class WebSocketLike(Protocol):
    """Small WebSocket surface required by the session loop."""

    async def receive_text(self) -> str: ...

    async def send_text(self, data: str) -> None: ...

    async def close(self) -> None: ...


PortFactory = Callable[[], Any]
_VAD_FRAME_BYTES = 768 * 2
_MAX_SPEECH_AUDIO_BYTES = 60 * 24_000 * 2
_MAX_HISTORY_TURNS = 20
_MAX_PENDING_TRANSCRIPT_WORDS = 500
_OUTPUT_AUDIO_CHUNK_BYTES = 4_800
_USAGE_UPDATE_INTERVAL_SECONDS = 60
_DEFAULT_VOICE_INSTRUCTIONS = (
    "You are a helpful, natural voice conversation partner. Answer the user's latest meaning, "
    "including corrections, using conversation history for context. Speak plainly and warmly, "
    "usually in one or two concise sentences. Avoid headings, bullets, markdown, long lists, and "
    "needless repetition. Ask a brief clarifying question only when needed to do the requested task. "
    "Do not narrate that you are thinking or listening, and never claim an action succeeded unless "
    "a tool confirms it. When a requested action is supported by an available tool, call the tool; "
    "do not merely restate the request or announce a proposed action as if it happened. Treat tool calls "
    "as actions, not topics to elaborate on: after a tool returns, "
    "state only the useful outcome and stop. Do not append stock follow-up offers, generic questions, "
    "or unsolicited suggestions. When a tool starts background work, acknowledge that it started once; "
    "do not promise or narrate later notifications because the runtime handles their delivery. Background "
    "results may be silent, deferred, or delivered after a sentence by the runtime; do not invent or "
    "repeat a separate notification."
)


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
    speech_audio: bytearray = field(default_factory=bytearray)
    output_audio_ms: int = 0
    speech_start_ms: int | None = None
    vad_pending_audio: bytearray = field(default_factory=bytearray)
    voice: str = "marin"
    delegation_ids: set[str] = field(default_factory=set)
    delegation_requests: dict[str, str] = field(default_factory=dict)
    pending_transcripts: list[str] = field(default_factory=list)


class LiveSession:
    """Run one authenticated WebSocket session until it closes."""

    def __init__(self, websocket: WebSocketLike, settings: Settings, port_factory: PortFactory) -> None:
        self.websocket = websocket
        self.settings = settings
        self.port_factory = port_factory
        self.state = SessionState()
        self.log = logger.bind(component="live-session", session_id=self.state.session_id)
        self.ports: Ports | None = None
        self.started_at = 0.0
        self._speech_queue: asyncio.PriorityQueue[tuple[int, int, str, bool]] = asyncio.PriorityQueue()
        self._speech_order = count()
        self._utterance_task: asyncio.Task[None] | None = None
        self._turn_tasks: set[asyncio.Task[None]] = set()
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._reply_task: asyncio.Task[str] | None = None
        self._pending_assistant_reply: str | None = None
        self._waiting_for_background_input = False
        self._background_input_ready = asyncio.Event()
        self._background_input_ready.set()
        self._routing_lock = asyncio.Lock()
        self._can_speak = asyncio.Event()
        self._can_speak.set()
        self._send_lock = asyncio.Lock()
        self._synthesis_cancel: ThreadEvent | None = None
        self._next_usage_update = 0.0
        self._tools: SessionTools | None = None

    async def run(self) -> None:
        """Read, validate, route, and acknowledge events until disconnect."""
        try:
            await self._run()
        finally:
            if not self.state.closing:
                await self.close("connection_lost")

    async def _run(self) -> None:
        """Run the event loop; ``run`` owns cleanup for every exit path."""
        while not self.state.closing:
            if self.state.started:
                now = time.monotonic()
                if now >= self.started_at + self.settings.max_session_duration_s:
                    await self.close("expired")
                    return
                if now >= self._next_usage_update:
                    await self._send_usage()
                    self._schedule_next_usage_update(now)
                    continue
            try:
                timeout = None
                if self.state.started:
                    timeout = min(
                        self.started_at + self.settings.max_session_duration_s,
                        self._next_usage_update,
                    ) - time.monotonic()
                raw = await asyncio.wait_for(self.websocket.receive_text(), timeout)
            except asyncio.TimeoutError:
                now = time.monotonic()
                if now >= self.started_at + self.settings.max_session_duration_s:
                    await self.close("expired")
                    return
                await self._send_usage()
                self._schedule_next_usage_update(now)
                continue
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
            try:
                remaining = self.started_at + self.settings.max_session_duration_s - time.monotonic()
                await asyncio.wait_for(self._handle(event), timeout=max(0.0, remaining))
            except asyncio.TimeoutError:
                await self.close("expired")
                return

    async def _start(self, event: ClientEvent) -> None:
        """Create this session's ports and announce the resolved session."""
        try:
            self.ports = self.port_factory()
        except Exception:
            self.log.exception("Session adapter initialization failed")
            await self._send_error("internal_error", "Unable to initialize session adapters")

        voice = event.data.get("voice")
        if isinstance(voice, str) and voice:
            self.state.voice = voice

        self.state.started = True
        self._tools = SessionTools(
            self._notify_timer,
            on_timer_started=self._timer_started,
            session_id=self.state.session_id,
        )
        self.log.info("Live session started", model=event.data["model"], voice=self.state.voice)
        self.started_at = time.monotonic()
        self._next_usage_update = self.started_at + _USAGE_UPDATE_INTERVAL_SECONDS
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
        self.log.debug("Handling client event", event_type=event.type, client_event_id=event.event_id)
        try:
            if event.type == "session.input_audio.append":
                if not self.state.muted:
                    await self._append_audio(event.data["audio"])
                return
            if event.type == "session.input_audio.mute":
                self.state.muted = True
                self._cancel_speech()
                self.state.vad_pending_audio.clear()
                self.state.speech_audio.clear()
                self.state.speech_start_ms = None
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
                if event.type == "session.commentary.append":
                    self._queue_speech(text)
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
            self.log.exception("Client event handling failed", event_type=event.type)
            await self._send_error("internal_error", "Session adapter failed")

    async def _append_audio(self, encoded_audio: str) -> None:
        """Decode one client audio event and finish turns at VAD boundaries."""
        try:
            pcm16le = b64decode(encoded_audio, validate=True)
        except (Base64Error, ValueError) as exc:
            raise ValueError("audio must be valid base64") from exc
        if self.ports is None:
            return
        self.state.vad_pending_audio.extend(pcm16le)
        while len(self.state.vad_pending_audio) >= _VAD_FRAME_BYTES:
            frame = bytes(self.state.vad_pending_audio[:_VAD_FRAME_BYTES])
            del self.state.vad_pending_audio[:_VAD_FRAME_BYTES]
            events = await asyncio.to_thread(self.ports.vad.append_audio, frame)
            self.log.debug("VAD frame processed", frame_bytes=len(frame), event_count=len(events))
            stopped_at: int | None = None
            for event in events:
                if event.type == "speech_started":
                    self.state.speech_start_ms = event.offset_ms
                    self._can_speak.clear()
                    self.log.info("User speech started; pausing assistant audio until routed", offset_ms=event.offset_ms)
                elif event.type == "speech_stopped":
                    self.log.info("User speech stopped", offset_ms=event.offset_ms)
                    stopped_at = event.offset_ms
            if self.state.speech_start_ms is not None:
                self.state.speech_audio.extend(frame)
                if len(self.state.speech_audio) > _MAX_SPEECH_AUDIO_BYTES:
                    excess = len(self.state.speech_audio) - _MAX_SPEECH_AUDIO_BYTES
                    del self.state.speech_audio[:excess]
                    self.state.speech_start_ms += excess // 48
            if stopped_at is not None:
                await self._finish_turn(stopped_at)

    async def _finish_turn(self, end_ms: int) -> None:
        """Move one completed audio buffer into background turn processing."""
        if not self.state.speech_audio:
            self.state.speech_start_ms = None
            return
        audio = bytes(self.state.speech_audio)
        start_ms = self.state.speech_start_ms or 0
        self.state.speech_audio.clear()
        self.state.speech_start_ms = None
        turn_id = uuid4().hex[:12]
        task = asyncio.create_task(self._process_turn(audio, start_ms, end_ms, turn_id))
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _process_turn(self, audio: bytes, start_ms: int, end_ms: int, turn_id: str) -> None:
        """Transcribe and route a turn without blocking microphone reception."""
        with logger.contextualize(turn_id=turn_id):
            started = time.perf_counter()
            self.log.info("Turn processing started", audio_bytes=len(audio), start_ms=start_ms, end_ms=end_ms)
            try:
                asr_started = time.perf_counter()
                transcript = await asyncio.to_thread(self.ports.asr.transcribe, audio, start_ms)
                self.log.info(
                    "ASR completed",
                    elapsed_ms=round((time.perf_counter() - asr_started) * 1000),
                    transcript_chars=len(transcript.text),
                )
                if not transcript.text:
                    self.log.info("Empty ASR result; no routing performed")
                    self._resume_speech()
                    return
                await self._send(
                    ServerEvent(
                        "session.input_transcript.delta",
                        {
                            "text": transcript.text,
                            "start_ms": transcript.start_ms,
                            "end_ms": min(transcript.end_ms, end_ms),
                        },
                    )
                )
                await self.handle_transcript(transcript.text)
            except asyncio.CancelledError:
                self.log.info("Turn processing cancelled")
                raise
            except Exception:
                self.log.exception("Turn processing failed")
                await self._send_error("internal_error", "Speech recognition failed")
                self._resume_speech()
            finally:
                self.log.info("Turn processing finished", elapsed_ms=round((time.perf_counter() - started) * 1000))

    async def handle_transcript(self, transcript: str) -> None:
        """Route one completed ASR transcript through the session adapters."""
        if not transcript.strip() or self.ports is None:
            return
        was_waiting_for_background_input = self._waiting_for_background_input
        if was_waiting_for_background_input:
            self._waiting_for_background_input = False
            self._background_input_ready.set()
            self.log.info("User responded to a background clarification")
            self._cancel_response()
            self._cancel_speech()
            self._pending_assistant_reply = None
            await self._send(ServerEvent("session.output_audio.cancelled"))
        try:
            async with self._routing_lock:
                combined_transcript = " ".join((*self.state.pending_transcripts, transcript))
                self.log.info(
                    "Routing transcript",
                    transcript_chars=len(combined_transcript),
                    pending_fragments=len(self.state.pending_transcripts),
                )
                routing_started = time.perf_counter()
                decision = await asyncio.to_thread(self.ports.router.classify, combined_transcript)
                self.log.info(
                    "Router decision",
                    elapsed_ms=round((time.perf_counter() - routing_started) * 1000),
                    kind=decision.kind,
                    talk=decision.talk,
                    task=decision.task,
                    interrupt_current=decision.interrupt_current,
                    wait_for_user=decision.wait_for_user,
                    labels=dict(decision.labels),
                )
                if decision.wait_for_user and len(combined_transcript.split()) <= _MAX_PENDING_TRANSCRIPT_WORDS:
                    self.state.pending_transcripts.append(transcript)
                    self.log.info(
                        "Waiting for user continuation",
                        pending_fragments=len(self.state.pending_transcripts),
                        pending_words=len(combined_transcript.split()),
                    )
                    self._resume_speech()
                    return
                self.state.pending_transcripts.clear()
                transcript = combined_transcript
            if decision.interrupt_current and not was_waiting_for_background_input:
                self.log.info("GLiNER classified a barge-in; cancelling active response")
                self._cancel_response()
                self._cancel_speech()
                await self._send(ServerEvent("session.output_audio.cancelled"))
                self._pending_assistant_reply = None
            else:
                if self._pending_assistant_reply is not None:
                    self.state.history.append(TranscriptTurn("assistant", self._pending_assistant_reply))
                    del self.state.history[:-_MAX_HISTORY_TURNS]
                    self._pending_assistant_reply = None
                active_reply = self._reply_task
                if active_reply is not None and not active_reply.done():
                    self._resume_speech()
                    await active_reply
                    if self._pending_assistant_reply is not None:
                        self.state.history.append(TranscriptTurn("assistant", self._pending_assistant_reply))
                        del self.state.history[:-_MAX_HISTORY_TURNS]
                        self._pending_assistant_reply = None
            self._resume_speech()
            if decision.talk:
                self.log.info("Starting LM Studio reply", transcript_chars=len(transcript), tool_count=len(self._tools.schemas) if self._tools else 0)
                request = TalkRequest(
                    transcript=transcript,
                    instructions=(_DEFAULT_VOICE_INSTRUCTIONS, *self.state.instructions),
                    thinking=tuple(self.state.thinking),
                    history=tuple(self.state.history),
                    tools=self._tools,
                )
                if self._tools is not None:
                    self._tools.set_request_context(transcript)
                self.state.history.append(TranscriptTurn("user", transcript))
                del self.state.history[:-_MAX_HISTORY_TURNS]
                reply_task = asyncio.create_task(self._reply(request))
                self._reply_task = reply_task
                try:
                    reply = await reply_task
                except asyncio.CancelledError:
                    self.log.info("LM Studio reply task cancelled")
                    return
                finally:
                    if self._reply_task is reply_task:
                        self._reply_task = None
                self._pending_assistant_reply = reply
                self.state.transcripts.append(transcript)
                self.log.info("LM Studio reply completed", reply_chars=len(reply))
            else:
                self.log.info("Talker skipped by router", task=decision.task, kind=decision.kind)
                self.state.history.append(TranscriptTurn("user", transcript))
                del self.state.history[:-_MAX_HISTORY_TURNS]
                self.state.transcripts.append(transcript)
            if decision.task:
                delegation_id = f"item_{uuid4().hex}"
                self.state.delegation_ids.add(delegation_id)
                self.state.delegation_requests[delegation_id] = transcript
                try:
                    await self._create_task(transcript, decision, delegation_id)
                except Exception:
                    self.state.delegation_ids.discard(delegation_id)
                    self.state.delegation_requests.pop(delegation_id, None)
                    self.log.exception("Task dispatch failed", delegation_id=delegation_id)
                    await self._send_error("task_dispatch_failed", "Task service is unavailable")
                else:
                    await self._send(
                        ServerEvent(
                            "session.delegation.created",
                            {"delegation_id": delegation_id, "target": "client"},
                        )
                    )
        except Exception:
            self.log.exception("Transcript routing failed")
            await self._send_error("internal_error", "Session adapter failed")
            self._resume_speech()

    def _cancel_response(self) -> None:
        """Stop the active model stream so no more stale sentences are queued."""
        task = self._reply_task
        self._reply_task = None
        if task is not None and not task.done():
            self.log.info("Cancelling active LLM response")
            task.cancel()

    def _resume_speech(self) -> None:
        """Allow buffered assistant speech to continue after routing completes."""
        self._can_speak.set()
        if not self._speech_queue.empty() and (self._utterance_task is None or self._utterance_task.done()):
            self._utterance_task = asyncio.create_task(self._drain_speech_queue())

    async def _reply(self, request: TalkRequest) -> str:
        """Stream text immediately and queue speech in small chunks as it arrives."""
        started = time.perf_counter()
        first_fragment = True
        stream_reply = getattr(self.ports.talker, "stream_reply", None)
        if stream_reply is None:
            reply = await self.ports.talker.reply(request)
            self.log.info("LLM reply received", elapsed_ms=round((time.perf_counter() - started) * 1000), reply_chars=len(reply))
            await self._send(ServerEvent("session.output_transcript.delta", {"text": reply}))
            self._queue_speech(reply, transcript_sent=True)
            return reply
        fragments: list[str] = []
        speech_buffer = ""
        async for fragment in stream_reply(request):
            if first_fragment:
                self.log.info("LLM first token received", elapsed_ms=round((time.perf_counter() - started) * 1000))
                first_fragment = False
            fragments.append(fragment)
            await self._send(ServerEvent("session.output_transcript.delta", {"text": fragment}))
            speech_buffer += fragment
            chunks, speech_buffer = _take_speech_chunks(speech_buffer)
            for chunk in chunks:
                self._queue_speech(
                    chunk,
                    transcript_sent=True,
                    priority=bool(getattr(request.tools, "urgent_tool_called", False)),
                )
        chunks, _ = _take_speech_chunks(speech_buffer, final=True)
        for chunk in chunks:
            self._queue_speech(
                chunk,
                transcript_sent=True,
                priority=bool(getattr(request.tools, "urgent_tool_called", False)),
            )
        reply = "".join(fragments)
        self.log.info(
            "LLM stream completed",
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            reply_chars=len(reply),
            fragments=len(fragments),
        )
        return reply

    def handle_task_result(self, delegation_id: str, content: str) -> bool:
        """Queue speech for a callback belonging to this session."""
        if self.state.closing or delegation_id not in self.state.delegation_ids or not content.strip():
            return False
        self.state.delegation_ids.remove(delegation_id)
        request = self.state.delegation_requests.pop(delegation_id, "")
        self._spawn_background_delivery(request, content)
        return True

    def _notify_timer(self, content: str, request: str, profile: ToolProfile) -> bool:
        """Deliver a completed timer to the live speech queue."""
        if self.state.closing:
            return False
        self._spawn_background_delivery(request, content, profile)
        return True

    def _spawn_background_delivery(
        self, request: str, content: str, profile: ToolProfile | None = None
    ) -> None:
        task = asyncio.create_task(
            self._deliver_background_result(request, content, profile)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _deliver_background_result(
        self, request: str, content: str, profile: ToolProfile | None = None
    ) -> None:
        """Classify and deliver a real background result without losing it."""
        if self.state.closing:
            return
        delivery = "after_playback"
        classifier = getattr(getattr(self.ports, "router", None), "classify_background_delivery", None)
        if callable(classifier):
            try:
                delivery = await asyncio.to_thread(classifier, request, content, profile)
            except Exception:
                self.log.exception("Background result classification failed; deferring speech")
        if self.state.closing:
            return
        if delivery not in {"silent", "after_playback", "interrupt_after_sentence", "wait_for_user"}:
            delivery = "after_playback"
        if profile is not None and profile.urgency == "urgent":
            delivery = "interrupt_after_sentence"
        self.log.info(
            "Background result delivery classified",
            delivery=delivery,
            tool_execution=profile.execution if profile is not None else None,
            tool_urgency=profile.urgency if profile is not None else None,
        )
        if delivery == "silent":
            self.state.commentary.append(content)
            await self._send(
                ServerEvent("session.background.update", {"message": "Background work completed quietly."})
            )
            return
        if delivery in {"interrupt_after_sentence", "wait_for_user"}:
            if delivery == "wait_for_user":
                self._waiting_for_background_input = True
                self._background_input_ready.clear()
                self.state.history.append(TranscriptTurn("assistant", content))
                del self.state.history[:-_MAX_HISTORY_TURNS]
                message = "I need your instructions before I continue."
            else:
                message = "Important background result; pausing after this sentence."
            await self._send(
                ServerEvent("session.background.update", {"message": message})
            )
            await self._send(
                ServerEvent(
                    "session.output_audio.interrupt",
                    {"after_sentence": True, "await_user": delivery == "wait_for_user"},
                )
            )
        else:
            await self._send(
                ServerEvent("session.background.update", {"message": "Background result ready; I’ll tell you when I finish speaking."})
            )
        self.state.commentary.append(content)
        if not self._queue_speech(
            content, priority=delivery in {"interrupt_after_sentence", "wait_for_user"}
        ):
            if delivery in {"interrupt_after_sentence", "wait_for_user"}:
                self._waiting_for_background_input = False
                await self._send(ServerEvent("session.output_audio.interrupt_aborted"))

    def _timer_started(self, seconds: int, message: str) -> None:
        """Tell the client that a real background timer has been scheduled."""
        self._announce_background(f"I’ll remind you in {seconds} seconds: {message}")

    def _announce_background(self, message: str) -> None:
        if self.state.closing:
            return
        asyncio.create_task(
            self._send(ServerEvent("session.background.update", {"message": message}))
        )

    def _queue_speech(
        self, text: str, *, transcript_sent: bool = False, priority: bool = False
    ) -> bool:
        """Queue one utterance; the worker synthesizes and sends it serially."""
        speaker = getattr(self.ports, "speaker", None) if self.ports is not None else None
        if not text.strip() or not callable(getattr(speaker, "synthesize", None)):
            return False
        self._speech_queue.put_nowait((0 if priority else 1, next(self._speech_order), text, transcript_sent))
        self.log.debug("Speech queued", text_chars=len(text), queue_size=self._speech_queue.qsize())
        if self._utterance_task is None or self._utterance_task.done():
            self._utterance_task = asyncio.create_task(self._drain_speech_queue())
        return True

    def _cancel_speech(self) -> None:
        """Cancel playback and discard queued utterances on barge-in."""
        self.log.debug(
            "Cancelling speech queue",
            queue_size=self._speech_queue.qsize(),
            synthesis_active=self._synthesis_cancel is not None,
        )
        if self._synthesis_cancel is not None:
            self._synthesis_cancel.set()
        if self._utterance_task is not None and not self._utterance_task.done():
            self._utterance_task.cancel()
        self._utterance_task = None
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def _drain_speech_queue(self) -> None:
        """Synthesize queued text and emit timed 24 kHz PCM chunks."""
        while not self._speech_queue.empty():
            await self._can_speak.wait()
            priority, _, text, transcript_sent = await self._next_speech()
            cancellation: ThreadEvent | None = None
            try:
                if not transcript_sent:
                    await self._send(ServerEvent("session.output_transcript.delta", {"text": text}))
                cancellation = ThreadEvent()
                self._synthesis_cancel = cancellation
                synthesis_started = time.perf_counter()
                self.log.info(
                    "TTS synthesis started",
                    text_chars=len(text),
                    priority=priority == 0,
                    queue_size=self._speech_queue.qsize(),
                )
                await self._send(
                    ServerEvent("session.output_audio.started", {"text": text, "priority": priority == 0})
                )
                audio = await asyncio.to_thread(
                    self.ports.speaker.synthesize, text, self.state.voice, cancellation
                )
                self.log.info(
                    "TTS synthesis completed",
                    elapsed_ms=round((time.perf_counter() - synthesis_started) * 1000),
                    audio_bytes=len(audio),
                    cancelled=cancellation.is_set(),
                )
                start_ms = self.state.output_audio_ms
                end_ms = start_ms + len(audio) // 48
                await self._can_speak.wait()
                await self._send(
                    ServerEvent(
                        "session.output_audio.transcript",
                        {"text": text, "start_ms": start_ms, "end_ms": end_ms, "priority": priority == 0},
                    )
                )
                for offset in range(0, len(audio), _OUTPUT_AUDIO_CHUNK_BYTES):
                    await self._can_speak.wait()
                    chunk = audio[offset : offset + _OUTPUT_AUDIO_CHUNK_BYTES]
                    start_ms = self.state.output_audio_ms
                    end_ms = start_ms + len(chunk) // 48
                    await self._send(
                        ServerEvent(
                            "session.output_audio.delta",
                            {
                                "delta": b64encode(chunk).decode("ascii"),
                                "start_ms": start_ms,
                                "end_ms": end_ms,
                                "priority": priority == 0,
                            },
                        )
                    )
                    self.log.debug("TTS audio chunk sent", audio_bytes=len(chunk), start_ms=start_ms, end_ms=end_ms)
                    self.state.output_audio_ms = end_ms
            except Exception:
                self.log.exception("TTS synthesis or delivery failed")
                await self._send_error("internal_error", "Speaker adapter failed")
                if priority == 0:
                    await self._send(ServerEvent("session.output_audio.interrupt_aborted"))
            finally:
                if cancellation is not None and self._synthesis_cancel is cancellation:
                    self._synthesis_cancel = None
                self._speech_queue.task_done()

    async def _next_speech(self) -> tuple[int, int, str, bool]:
        """Hold ordinary reply speech while a background result awaits user input."""
        while self._waiting_for_background_input:
            try:
                item = self._speech_queue.get_nowait()
            except asyncio.QueueEmpty:
                await self._background_input_ready.wait()
                continue
            if item[0] == 0:
                return item
            self._speech_queue.task_done()
            self._speech_queue.put_nowait(item)
            await self._background_input_ready.wait()
        return await self._speech_queue.get()

    async def _create_task(self, transcript: str, decision: RouteDecision, delegation_id: str) -> None:
        """Create one outbound task while preserving the router labels."""
        method = getattr(self.ports.tasks, "create", None)
        if method is None:
            raise RuntimeError("task adapter is not configured")
        result = method(delegation_id, self.state.session_id, transcript, dict(decision.labels))
        if inspect.isawaitable(result):
            await result

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
        async with self._send_lock:
            await self.websocket.send_text(event.to_json())

    async def close(self, reason: str) -> None:
        """Emit final usage once and close the underlying WebSocket."""
        if self.state.closing:
            return
        self.state.closing = True
        self.log.info("Closing live session", reason=reason)
        if self._tools is not None:
            await self._tools.close()
            self._tools = None
        self._can_speak.set()
        self._waiting_for_background_input = False
        self._background_input_ready.set()
        self._cancel_response()
        turn_tasks = tuple(self._turn_tasks)
        for task in turn_tasks:
            task.cancel()
        background_tasks = tuple(self._background_tasks)
        for task in background_tasks:
            task.cancel()
        utterance_task = self._utterance_task
        self._cancel_speech()
        await asyncio.gather(*turn_tasks, return_exceptions=True)
        await asyncio.gather(*background_tasks, return_exceptions=True)
        if utterance_task is not None:
            await asyncio.gather(utterance_task, return_exceptions=True)
        try:
            await self._send(
                ServerEvent(
                    "session.closed",
                    {"usage": {"seconds": self._usage_seconds()}, "reason": reason},
                )
            )
        except Exception:
            pass
        try:
            await self.websocket.close()
        except Exception:
            pass
        try:
            await self._close_ports()
        finally:
            self.ports = None

    def _usage_seconds(self) -> int:
        """Return cumulative wall-clock time since session.start."""
        if not self.state.started:
            return 0
        return int(time.monotonic() - self.started_at)

    async def _send_usage(self) -> None:
        """Report cumulative wall-clock usage and the unused context window."""
        await self._send(
            ServerEvent(
                "session.usage.updated",
                {"usage": {"seconds": self._usage_seconds()}, "context_window": {"usage_ratio": 0.0}},
            )
        )

    def _schedule_next_usage_update(self, now: float) -> None:
        """Schedule the next whole-minute usage update without timer drift."""
        elapsed = now - self.started_at
        self._next_usage_update = self.started_at + (
            (int(elapsed // _USAGE_UPDATE_INTERVAL_SECONDS) + 1) * _USAGE_UPDATE_INTERVAL_SECONDS
        )

    async def _close_ports(self) -> None:
        """Close unique adapter resources when they expose a close hook."""
        if self.ports is None:
            return
        closed: set[int] = set()
        # Only VAD and ASR are session-scoped; shared ports and posted tasks outlive it.
        for name in ("vad", "asr"):
            port = getattr(self.ports, name, None)
            if port is None or id(port) in closed:
                continue
            closed.add(id(port))
            close = getattr(port, "close", None)
            if not callable(close):
                continue
            try:
                if inspect.iscoroutinefunction(close):
                    await close()
                else:
                    result = await asyncio.to_thread(close)
                    if inspect.isawaitable(result):
                        await result
            except Exception:
                continue


def _take_speech_chunks(text: str, *, final: bool = False) -> tuple[list[str], str]:
    """Extract complete sentences for speech; leave partial text buffered."""
    chunks: list[str] = []
    remaining = text
    while remaining:
        boundary = re.search(r"[.!?](?:[\"')\]]*)\s", remaining)
        if boundary is not None:
            chunk = remaining[: boundary.end()].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[boundary.end() :].lstrip()
            continue
        break
    if final and remaining.strip():
        chunks.append(remaining.strip())
        remaining = ""
    return chunks, remaining
