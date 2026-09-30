import os
import unittest
from unittest.mock import patch

from demo.client import endpoint_url, upstream_settings


class EndpointSettingsTests(unittest.TestCase):
    def test_endpoint_url_adds_live_route_and_converts_scheme(self) -> None:
        self.assertEqual(endpoint_url("http://localhost:8000"), "ws://localhost:8000/v1/live/sessions")
        self.assertEqual(endpoint_url("https://example.test/v1/live/sessions/"), "wss://example.test/v1/live/sessions")

    def test_bridge_uses_demo_credentials_without_leaking_them_to_ui(self) -> None:
        env = {
            "GPTLIVE_DEMO_WS_URL": "ws://localhost:8000",
            "GPTLIVE_ENDPOINT_URL": "",
            "GPTLIVE_DEMO_TOKEN": "demo-secret",
            "GPTLIVE_BEARER_TOKEN": "",
        }
        with patch.dict(os.environ, env):
            endpoint, token = upstream_settings()

        self.assertEqual(endpoint, "ws://localhost:8000/v1/live/sessions")
        self.assertEqual(token, "demo-secret")

    def test_server_environment_names_are_supported(self) -> None:
        env = {
            "GPTLIVE_DEMO_WS_URL": "",
            "GPTLIVE_ENDPOINT_URL": "http://localhost:8000",
            "GPTLIVE_DEMO_TOKEN": "",
            "GPTLIVE_BEARER_TOKEN": "server-secret",
        }
        with patch.dict(os.environ, env):
            endpoint, token = upstream_settings()

        self.assertEqual(endpoint, "ws://localhost:8000/v1/live/sessions")
        self.assertEqual(token, "server-secret")


if __name__ == "__main__":
    unittest.main()
