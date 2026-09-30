import json
import base64
import asyncio
from dataclasses import replace
from threading import Event
import unittest
from unittest.mock import patch

from open_gptlive_poc.live.session import LiveSession
from open_gptlive_poc.ports.router import RouteDecision
from open_gptlive_poc.ports.talker import TalkRequest


class FakeWebSocket:
    def __init__(self, *events: dict[str, object]) -> None:
        self.events = [json.dumps(event) for event in events]
        self.sent: list[dict[str, object]] = []
        self.closed = False

    async def receive_text(self) -> str:
        if not self.events:
            raise EOFError
        return self.events.pop(0)

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        self.closed = True


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_talker_reply_is_spoken_as_timed_audio_after_transcript(self) -> None:
        class Router:
            def classify(self, transcript):
                return RouteDecision(transcript, 0.0, "chat", True, {})

        class Talker:
            async def reply(self, request: TalkRequest) -> str:
                return "Hello"

        class Speaker:
            def synthesize(self, text: str, voice: str, cancel_event: Event | None = None) -> bytes:
                self.request = (text, voice)
                return b"\x01\x00" * 2_400

        websocket = FakeWebSocket()
        speaker = Speaker()
        session = LiveSession(websocket, self._settings(), lambda: None)
        session.ports = type(
            "Ports",
            (),
            {"router": Router(), "talker": Talker(), "speaker": speaker, "tasks": object(), "vad": object()},
        )()

        await session.handle_transcript("Hi")
        await session._utterance_task

        self.assertEqual(speaker.request, ("Hello", "marin"))
        self.assertEqual(
            [event["type"] for event in websocket.sent],
            ["session.output_transcript.delta", "session.output_audio.delta"],
        )
        self.assertEqual(len(base64.b64decode(websocket.sent[1]["delta"])), 4_800)
        self.assertEqual(websocket.sent[1]["end_ms"] - websocket.sent[1]["start_ms"], 100)

    async def test_barge_in_cancels_remaining_audio_chunks(self) -> None:
        class SlowWebSocket(FakeWebSocket):
            async def send_text(self, data: str) -> None:
                await super().send_text(data)
                await asyncio.sleep(0.01)

        class Speaker:
            def synthesize(self, text: str, voice: str, cancel_event: Event | None = None) -> bytes:
                return b"\x00\x00" * 24_000

        websocket = SlowWebSocket()
        session = LiveSession(websocket, self._settings(), lambda: None)
        session.ports = type("Ports", (), {"speaker": Speaker()})()
        session._queue_speech("Hello")
        utterance = session._utterance_task

        while not any(event["type"] == "session.output_audio.delta" for event in websocket.sent):
            await asyncio.sleep(0.001)
        session._cancel_speech()
        with self.assertRaises(asyncio.CancelledError):
            await utterance

        self.assertEqual(sum(event["type"] == "session.output_audio.delta" for event in websocket.sent), 1)

    async def test_cancelled_utterance_does_not_clear_next_cancellation_event(self) -> None:
        started = Event()
        release = Event()

        class Speaker:
            def synthesize(self, text: str, voice: str, cancel_event: Event | None = None) -> bytes:
                started.set()
                release.wait(timeout=2)
                return b"\x00\x00" * 2_400

        session = LiveSession(FakeWebSocket(), self._settings(), lambda: None)
        session.ports = type("Ports", (), {"speaker": Speaker()})()
        session._queue_speech("First")
        old_utterance = session._utterance_task
        await asyncio.to_thread(started.wait, 1)

        session._cancel_speech()
        next_cancellation = Event()
        session._synthesis_cancel = next_cancellation
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await old_utterance

        self.assertIs(session._synthesis_cancel, next_cancellation)

    async def test_task_result_content_is_queued_for_speech(self) -> None:
        class Speaker:
            def synthesize(self, text: str, voice: str, cancel_event: Event | None = None) -> bytes:
                self.request = (text, voice)
                return b"\x00\x00" * 2_400

        websocket = FakeWebSocket()
        speaker = Speaker()
        session = LiveSession(websocket, self._settings(), lambda: None)
        session.ports = type("Ports", (), {"speaker": speaker})()
        session.state.delegation_ids.add("item_1")

        self.assertTrue(session.handle_task_result("item_1", "Task complete"))
        await session._utterance_task

        self.assertEqual(speaker.request, ("Task complete", "marin"))
        self.assertEqual(websocket.sent[0]["type"], "session.output_transcript.delta")
        self.assertEqual(websocket.sent[1]["type"], "session.output_audio.delta")
        self.assertFalse(session.handle_task_result("unknown", "Ignored"))
        self.assertFalse(session.handle_task_result("item_1", "Repeated"))

    async def test_closing_session_rejects_task_callback(self) -> None:
        session = LiveSession(FakeWebSocket(), self._settings(), lambda: None)
        session.state.delegation_ids.add("item_1")
        session.state.closing = True

        self.assertFalse(session.handle_task_result("item_1", "Late result"))

    async def test_transcript_talk_task_and_both_paths(self) -> None:
        class Router:
            def __init__(self, decision):
                self.decision = decision

            def classify(self, transcript):
                return self.decision

        class Talker:
            async def reply(self, request: TalkRequest) -> str:
                self.request = request
                return "Sure."

        class Tasks:
            async def create(self, delegation_id, session_id, transcript, labels):
                self.request = (delegation_id, session_id, transcript, labels)

        talker = Talker()
        tasks = Tasks()
        websocket = FakeWebSocket()
        session = LiveSession(websocket, self._settings(), lambda: None)
        session.ports = type("Ports", (), {"router": Router(RouteDecision("", 0.9, "task", True, {"kind": "task"})), "talker": talker, "tasks": tasks, "vad": object()})()

        await session.handle_transcript("Do it")

        self.assertEqual(websocket.sent[0]["type"], "session.output_transcript.delta")
        self.assertEqual(websocket.sent[1]["type"], "session.delegation.created")
        self.assertEqual(tasks.request[2:], ("Do it", {"kind": "task"}))
        self.assertEqual(talker.request.transcript, "Do it")
        self.assertEqual(websocket.sent[1]["delegation_id"], tasks.request[0])

    async def test_transcript_adapter_failure_is_recoverable(self) -> None:
        class Router:
            def classify(self, transcript):
                raise RuntimeError("broken")

        websocket = FakeWebSocket()
        session = LiveSession(websocket, self._settings(), lambda: None)
        session.ports = type("Ports", (), {"router": Router(), "talker": object(), "tasks": object(), "vad": object()})()

        await session.handle_transcript("Hello")

        self.assertEqual(websocket.sent[0]["type"], "error")
        self.assertEqual(websocket.sent[0]["error"]["code"], "internal_error")

    async def test_task_http_failure_is_session_scoped(self) -> None:
        class Router:
            def classify(self, transcript):
                return RouteDecision(transcript, 0.9, "task", False, {"kind": "task"})

        class Tasks:
            async def create(self, delegation_id, session_id, transcript, labels):
                raise OSError("task service unavailable")

        websocket = FakeWebSocket()
        session = LiveSession(websocket, self._settings(), lambda: None)
        session.ports = type(
            "Ports",
            (),
            {"router": Router(), "tasks": Tasks(), "talker": object(), "speaker": object(), "vad": object()},
        )()

        await session.handle_transcript("Do it")

        self.assertEqual(websocket.sent[-1]["error"]["code"], "internal_error")
        self.assertFalse(session.state.delegation_ids)
        self.assertFalse(any(event["type"] == "session.delegation.created" for event in websocket.sent))

    async def test_start_commands_and_close(self) -> None:
        websocket = FakeWebSocket(
            {"type": "session.start", "event_id": "start", "model": "gpt-live-1"},
            {"type": "session.instructions.append", "event_id": "instructions", "text": "Be concise."},
            {"type": "session.input_audio.mute", "event_id": "mute"},
            {"type": "session.input_audio.unmute", "event_id": "unmute"},
            {"type": "session.close"},
        )
        session = LiveSession(websocket, self._settings(), lambda: None)

        await session.run()

        self.assertEqual(
            [event["type"] for event in websocket.sent],
            [
                "session.started",
                "session.instructions.appended",
                "session.input_audio.muted",
                "session.input_audio.unmuted",
                "session.closed",
            ],
        )
        self.assertEqual(websocket.sent[1]["client_event_id"], "instructions")
        self.assertEqual(websocket.sent[0]["client_event_id"], "start")
        self.assertTrue(websocket.closed)
        self.assertEqual(session.state.instructions, ["Be concise."])
        self.assertEqual(websocket.sent[-1]["reason"], "close_requested")
        self.assertEqual(websocket.sent[-1]["usage"]["seconds"], 0)

    async def test_usage_updates_and_expiry_close_resources(self) -> None:
        class WaitingWebSocket(FakeWebSocket):
            async def receive_text(self) -> str:
                if self.events:
                    return await super().receive_text()
                await asyncio.Future()

        class Resource:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        websocket = WaitingWebSocket({"type": "session.start", "model": "gpt-live-1"})
        resource = Resource()
        shared_speaker = Resource()
        task_adapter = Resource()
        session = LiveSession(
            websocket,
            replace(self._settings(), max_session_duration_s=0.12),
            lambda: type("Ports", (), {"vad": resource, "speaker": shared_speaker, "tasks": task_adapter})(),
        )

        with patch("open_gptlive_poc.live.session._USAGE_UPDATE_INTERVAL_SECONDS", 0.025):
            await session.run()

        updates = [event["usage"]["seconds"] for event in websocket.sent if event["type"] == "session.usage.updated"]
        self.assertGreaterEqual(len(updates), 2)
        self.assertEqual(updates, sorted(updates))
        self.assertEqual(websocket.sent[-1]["type"], "session.closed")
        self.assertEqual(websocket.sent[-1]["reason"], "expired")
        self.assertGreaterEqual(websocket.sent[-1]["usage"]["seconds"], updates[-1])
        self.assertTrue(resource.closed)
        self.assertFalse(shared_speaker.closed)
        self.assertFalse(task_adapter.closed)
        self.assertIsNone(session.ports)
        self.assertTrue(websocket.closed)

    async def test_expiry_interrupts_in_flight_session_work(self) -> None:
        operation_started = Event()
        release_operation = Event()

        class SlowVAD:
            def append_audio(self, frame: bytes):
                operation_started.set()
                release_operation.wait(timeout=1)
                return []

        websocket = FakeWebSocket(
            {"type": "session.start", "model": "gpt-live-1"},
            {"type": "session.input_audio.append", "audio": base64.b64encode(b"\x00\x00" * 768).decode()},
        )
        session = LiveSession(
            websocket,
            replace(self._settings(), max_session_duration_s=1),
            lambda: type("Ports", (), {"vad": SlowVAD()})(),
        )

        try:
            with patch("open_gptlive_poc.live.session._USAGE_UPDATE_INTERVAL_SECONDS", 60):
                await session.run()
            self.assertTrue(operation_started.is_set())
            self.assertEqual(websocket.sent[-1]["type"], "session.closed")
            self.assertEqual(websocket.sent[-1]["reason"], "expired")
        finally:
            release_operation.set()

    async def test_disconnect_closes_with_connection_lost_reason(self) -> None:
        websocket = FakeWebSocket({"type": "session.start", "model": "gpt-live-1"})
        session = LiveSession(websocket, self._settings(), lambda: None)

        await session.run()

        self.assertEqual(websocket.sent[-1]["type"], "session.closed")
        self.assertEqual(websocket.sent[-1]["reason"], "connection_lost")

    async def test_close_cancels_current_and_queued_speech(self) -> None:
        synthesis_started = Event()

        class Speaker:
            def __init__(self):
                self.requests = []

            def synthesize(self, text: str, voice: str, cancel_event: Event | None = None) -> bytes:
                self.requests.append(text)
                synthesis_started.set()
                while cancel_event is not None and not cancel_event.wait(0.01):
                    pass
                return b"\x00\x00" * 2_400

        websocket = FakeWebSocket()
        speaker = Speaker()
        session = LiveSession(websocket, self._settings(), lambda: None)
        session.ports = type("Ports", (), {"speaker": speaker})()
        session._queue_speech("Current")
        await asyncio.to_thread(synthesis_started.wait, 1)
        session._queue_speech("Queued")

        await session.close("close_requested")

        self.assertEqual(speaker.requests, ["Current"])
        self.assertEqual(websocket.sent[-1]["type"], "session.closed")
        self.assertFalse(any(event["type"] == "session.output_audio.delta" for event in websocket.sent))

    async def test_commands_before_start_are_rejected_and_session_stays_up(self) -> None:
        websocket = FakeWebSocket(
            {"type": "session.close"},
            {"type": "session.start", "model": "gpt-live-1"},
            {"type": "session.close"},
        )
        session = LiveSession(websocket, self._settings(), lambda: None)

        await session.run()

        self.assertEqual(websocket.sent[0]["type"], "error")
        self.assertEqual(websocket.sent[0]["error"]["code"], "invalid_value")
        self.assertEqual(websocket.sent[-1]["type"], "session.closed")

    async def test_each_session_gets_its_own_ports(self) -> None:
        created: list[object] = []

        def factory() -> object:
            port = object()
            created.append(port)
            return port

        first = LiveSession(FakeWebSocket({"type": "session.start", "model": "gpt-live-1"}, {"type": "session.close"}), self._settings(), factory)
        second = LiveSession(FakeWebSocket({"type": "session.start", "model": "gpt-live-1"}, {"type": "session.close"}), self._settings(), factory)

        await first.run()
        await second.run()

        self.assertEqual(len(created), 2)
        self.assertIsNot(created[0], created[1])

    @staticmethod
    def _settings():
        from open_gptlive_poc.config import Settings

        return Settings(
            bearer_token="test-token",
            silero_model_path="silero",
            whisper_model_path="whisper",
            gliner_model_path="gliner",
        )


if __name__ == "__main__":
    unittest.main()
