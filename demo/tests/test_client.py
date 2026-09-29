import unittest

import numpy as np

from demo.client import convert_to_pcm16le, endpoint_url


class DemoClientTests(unittest.TestCase):
    def test_audio_is_mono_24khz_pcm16le(self) -> None:
        samples = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)

        result = convert_to_pcm16le((48_000, samples))

        self.assertEqual(len(result) % 2, 0)
        self.assertGreater(len(result), 0)

    def test_integer_audio_is_scaled_from_pcm_range(self) -> None:
        result = convert_to_pcm16le((24_000, np.array([[0, 32767], [0, 32767]], dtype=np.int16)))

        self.assertEqual(np.frombuffer(result, dtype="<i2").tolist(), [16383, 16383])

    def test_endpoint_url_is_normalized(self) -> None:
        self.assertEqual(endpoint_url("https://example.com"), "wss://example.com/v1/live/sessions")
        self.assertEqual(endpoint_url("ws://example.com/v1/live/sessions"), "ws://example.com/v1/live/sessions")


if __name__ == "__main__":
    unittest.main()
