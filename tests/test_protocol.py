import base64
import json
import unittest

from open_gptlive_poc.live.protocol import ProtocolError, authenticate, parse_client_event


class ProtocolTests(unittest.TestCase):
    def test_start_without_audio_uses_fixed_audio_contract(self) -> None:
        event = parse_client_event(json.dumps({"type": "session.start", "model": "gpt-live-1"}))

        self.assertEqual(event.type, "session.start")

    def test_start_round_trips_and_ignores_extra_fields(self) -> None:
        event = parse_client_event(
            json.dumps(
                {
                    "type": "session.start",
                    "event_id": "evt_1",
                    "model": "gpt-live-1",
                    "delegation": {"type": "client"},
                    "audio": {"format": {"encoding": "pcm16le", "sample_rate": 24000, "channels": 1}},
                    "future_field": True,
                }
            )
        )

        self.assertEqual(event.type, "session.start")
        self.assertEqual(event.event_id, "evt_1")
        self.assertTrue(event.data["future_field"])

    def test_audio_requires_even_pcm16_bytes(self) -> None:
        encoded = base64.b64encode(b"odd").decode()

        with self.assertRaisesRegex(ProtocolError, "PCM16") as raised:
            parse_client_event(json.dumps({"type": "session.input_audio.append", "audio": encoded}))

        self.assertEqual(raised.exception.code, "invalid_audio")

    def test_supplied_audio_format_must_be_complete_and_exact(self) -> None:
        base = {"type": "session.start", "model": "gpt-live-1"}
        partial = {**base, "audio": {"format": {"sample_rate": 24000}}}
        conflicting = {**base, "audio": {"format": {"encoding": "pcm16", "sample_rate": 24000, "channels": 1}}}

        with self.assertRaisesRegex(ProtocolError, "encoding is required"):
            parse_client_event(json.dumps(partial))
        with self.assertRaisesRegex(ProtocolError, "must be pcm16le"):
            parse_client_event(json.dumps(conflicting))

    def test_immutable_update_includes_client_event_id(self) -> None:
        with self.assertRaises(ProtocolError) as raised:
            parse_client_event(
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "evt_update",
                        "session": {"delegation": {"type": "server"}},
                    }
                )
            )

        error = raised.exception.as_error().as_dict()
        self.assertEqual(error["error"]["code"], "immutable_field_update")
        self.assertEqual(error["error"]["client_event_id"], "evt_update")

    def test_invalid_json_is_a_recoverable_protocol_error(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "valid JSON") as raised:
            parse_client_event("not-json")

        self.assertEqual(raised.exception.as_error().as_dict()["type"], "error")

    def test_bearer_authentication_is_exact(self) -> None:
        self.assertTrue(authenticate("Bearer secret", "secret"))
        self.assertFalse(authenticate("Bearer wrong", "secret"))
        self.assertFalse(authenticate("Basic secret", "secret"))


if __name__ == "__main__":
    unittest.main()
