import unittest

from fastapi.testclient import TestClient

from open_gptlive_poc.app import create_app
from open_gptlive_poc.config import Settings


class AppTests(unittest.TestCase):
    def test_task_callback_authenticates_and_delivers_content(self) -> None:
        app = create_app(
            Settings(
                bearer_token="secret",
                silero_model_path="silero",
                whisper_model_path="whisper",
                gliner_model_path="gliner",
            )
        )
        deliveries: list[tuple[str, str]] = []

        class Session:
            def handle_task_result(self, delegation_id: str, content: str) -> bool:
                deliveries.append((delegation_id, content))
                return delegation_id == "item_1"

        app.state.sessions["session_1"] = Session()
        client = TestClient(app)

        unauthorized = client.post(
            "/internal/delegations/item_1/result",
            json={"content": "Done"},
        )
        accepted = client.post(
            "/internal/delegations/item_1/result",
            headers={"Authorization": "Bearer secret"},
            json={"content": "Done"},
        )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.json(), {"status": "accepted"})
        self.assertEqual(deliveries, [("item_1", "Done")])


if __name__ == "__main__":
    unittest.main()
