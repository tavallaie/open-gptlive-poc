import json
import base64
import asyncio
import unittest

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
            def synthesize(self, text: str, voice: str) -> bytes:
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
            def synthesize(self, text: str, voice: str) -> bytes:
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

    async def test_task_result_content_is_queued_for_speech(self) -> None:
        class Speaker:
            def synthesize(self, text: str, voice: str) -> bytes:
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
            async def create(self, session_id, transcript, labels):
                self.request = (session_id, transcript, labels)
                return "item_task"

        talker = Talker()
        tasks = Tasks()
        websocket = FakeWebSocket()
        session = LiveSession(websocket, self._settings(), lambda: None)
        session.ports = type("Ports", (), {"router": Router(RouteDecision("", 0.9, "task", True, {"kind": "task"})), "talker": talker, "tasks": tasks, "vad": object()})()

        await session.handle_transcript("Do it")

        self.assertEqual(websocket.sent[0]["type"], "session.output_transcript.delta")
        self.assertEqual(websocket.sent[1]["type"], "session.delegation.created")
        self.assertEqual(tasks.request[1:], ("Do it", {"kind": "task"}))
        self.assertEqual(talker.request.transcript, "Do it")

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
