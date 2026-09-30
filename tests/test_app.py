import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from open_gptlive_poc.app import create_app
from open_gptlive_poc.adapters.sqlite_delegations import SQLiteDelegations
from open_gptlive_poc.config import Settings


class AppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)

    def _settings(self) -> Settings:
        return Settings(
            bearer_token="secret",
            silero_model_path="silero",
            whisper_model_path="whisper",
            gliner_model_path="gliner",
            delegation_db_path=str(Path(self.temp_directory.name) / "delegations.sqlite3"),
        )

    def test_voice_demo_is_served_by_main_app(self) -> None:
        app = create_app(self._settings())
        with TestClient(app) as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Open GPT Live", response.text)
        self.assertIn('id="token"', response.text)

    def test_browser_websocket_authenticates_before_starting_session(self) -> None:
        app = create_app(self._settings())
        with TestClient(app) as client:
            with client.websocket_connect("/ws/live") as websocket:
                websocket.send_json({"type": "auth", "token": "secret"})
                self.assertEqual(websocket.receive_json(), {"type": "session.authenticated"})

    def test_browser_and_api_websockets_allow_empty_configured_token(self) -> None:
        app = create_app(replace(self._settings(), bearer_token=None))
        with TestClient(app) as client:
            with client.websocket_connect("/ws/live") as browser:
                browser.send_json({"type": "auth", "token": ""})
                self.assertEqual(browser.receive_json(), {"type": "session.authenticated"})
            with client.websocket_connect("/v1/live/sessions") as api:
                api.send_json({"type": "session.start", "model": "gpt-live-1"})
                self.assertEqual(api.receive_json()["type"], "session.started")

    def test_task_callback_authenticates_and_delivers_content(self) -> None:
        app = create_app(self._settings())
        deliveries: list[tuple[str, str]] = []

        class Session:
            def handle_task_result(self, delegation_id: str, content: str) -> bool:
                deliveries.append((delegation_id, content))
                return delegation_id == "item_1"

        app.state.sessions["session_1"] = Session()
        with TestClient(app) as client:
            app.state.delegations.register("item_1", "session_1")
            unauthorized = client.post(
                "/internal/delegations/item_1/result",
                json={"content": "Done"},
            )
            accepted = client.post(
                "/internal/delegations/item_1/result",
                headers={"Authorization": "Bearer secret"},
                json={"content": "Done"},
            )
            duplicate = client.post(
                "/internal/delegations/item_1/result",
                headers={"Authorization": "Bearer secret"},
                json={"content": "Again"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.json(), {"status": "accepted"})
        self.assertEqual(duplicate.status_code, 404)
        self.assertEqual(deliveries, [("item_1", "Done")])

    def test_callback_rejects_malformed_oversized_and_long_content(self) -> None:
        app = create_app(self._settings())
        client = TestClient(app)
        headers = {"Authorization": "Bearer secret"}

        malformed = client.post("/internal/delegations/item_1/result", headers=headers, content=b"{")
        oversized = client.post("/internal/delegations/item_1/result", headers=headers, content=b" " * 65_537)
        too_many_words = client.post(
            "/internal/delegations/item_1/result",
            headers=headers,
            json={"content": "word " * 501},
        )

        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(too_many_words.status_code, 422)

    def test_callback_is_delivered_to_another_worker_app(self) -> None:
        settings = self._settings()
        owner_store = SQLiteDelegations(settings.delegation_db_path)
        callback_store = SQLiteDelegations(settings.delegation_db_path)
        owner_app = create_app(settings, delegations=owner_store)
        callback_app = create_app(settings, delegations=callback_store)
        deliveries: list[tuple[str, str]] = []

        class Session:
            def handle_task_result(self, delegation_id: str, content: str) -> bool:
                deliveries.append((delegation_id, content))
                return delegation_id == "item_cross_worker"

        with TestClient(owner_app) as owner_client:
            owner_store.register("item_cross_worker", "session_owner")
            owner_app.state.sessions["session_owner"] = Session()
            with TestClient(callback_app) as callback_client:
                response = callback_client.post(
                    "/internal/delegations/item_cross_worker/result",
                    headers={"Authorization": "Bearer secret"},
                    json={"content": "Completed on the owning worker"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(deliveries, [("item_cross_worker", "Completed on the owning worker")])

    def test_accepted_callback_is_not_delivered_twice_if_ack_write_retries(self) -> None:
        settings = self._settings()

        class FailingOnceStore(SQLiteDelegations):
            fail_next_completion = True

            def complete(self, callback_id: str, accepted: bool) -> None:
                if self.fail_next_completion:
                    self.fail_next_completion = False
                    raise sqlite3.OperationalError("temporary database lock")
                super().complete(callback_id, accepted)

        store = FailingOnceStore(settings.delegation_db_path)
        app = create_app(settings, delegations=store)
        deliveries: list[tuple[str, str]] = []

        class Session:
            def handle_task_result(self, delegation_id: str, content: str) -> bool:
                deliveries.append((delegation_id, content))
                return True

        with TestClient(app) as client:
            store.register("item_retry_ack", "session_1")
            app.state.sessions["session_1"] = Session()
            with self.assertLogs("open_gptlive_poc.app", level="ERROR"):
                response = client.post(
                    "/internal/delegations/item_retry_ack/result",
                    headers={"Authorization": "Bearer secret"},
                    json={"content": "Once only"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(deliveries, [("item_retry_ack", "Once only")])


if __name__ == "__main__":
    unittest.main()
