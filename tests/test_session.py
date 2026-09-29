import json
import unittest

from open_gptlive_poc.live.session import LiveSession


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
