import unittest
from unittest.mock import patch

from open_gptlive_poc.config import ConfigurationError, Settings


VALID_ENV = {
    "GPTLIVE_BEARER_TOKEN": "test-token",
    "GPTLIVE_SILERO_MODEL_PATH": "/models/silero.onnx",
    "GPTLIVE_WHISPER_MODEL_PATH": "/models/whisper",
    "GPTLIVE_GLINER_MODEL_PATH": "/models/GLiNER2.5-Decide",
}


class SettingsTests(unittest.TestCase):
    def test_defaults_are_applied(self) -> None:
        settings = Settings.from_env(VALID_ENV)

        self.assertEqual(settings.port, 8000)
        self.assertEqual(settings.vad_pause_ms, 700)
        self.assertEqual(settings.supertonic_voice_map, {"marin": "F1"})

    def test_missing_required_values_are_reported(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "GPTLIVE_BEARER_TOKEN"):
            Settings.from_env({})

    def test_empty_environment_mapping_does_not_use_process_environment(self) -> None:
        with patch.dict("os.environ", VALID_ENV):
            with self.assertRaisesRegex(ConfigurationError, "GPTLIVE_BEARER_TOKEN"):
                Settings.from_env({})

    def test_invalid_threshold_is_rejected(self) -> None:
        environment = {**VALID_ENV, "GPTLIVE_GLINER_TASK_THRESHOLD": "2"}

        with self.assertRaisesRegex(ConfigurationError, "between 0 and 1"):
            Settings.from_env(environment)


if __name__ == "__main__":
    unittest.main()
