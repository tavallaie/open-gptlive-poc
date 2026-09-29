import json
import unittest

from open_gptlive_poc.adapters.gliner_router import GLiNERRouter
from open_gptlive_poc.adapters.lmstudio_talker import LMStudioTalker
from open_gptlive_poc.ports.talker import TalkRequest, TranscriptTurn


class FakeGLiNER:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def classify_text(self, transcript, questions):
        self.calls.append((transcript, questions))
        return self.result


class RouterTests(unittest.TestCase):
    def test_task_decision_preserves_labels_and_threshold(self):
        model = FakeGLiNER(
            {
                "needs_task": {"task": 0.8, "no_task": 0.2},
                "kind": {"task": 0.9, "chat": 0.1},
                "response": {"both": 0.9, "talk": 0.1},
            }
        )
        router = GLiNERRouter("unused", model=model, task_threshold=0.75)

        decision = router.classify("Where is my order?")

        self.assertTrue(decision.task)
        self.assertTrue(decision.talk)
        self.assertEqual(decision.kind, "task")
        self.assertEqual(decision.labels["needs_task"], {"task": 0.8, "no_task": 0.2})
        self.assertEqual(model.calls[0][0], "Where is my order?")

    def test_chat_decision_does_not_schedule_task(self):
        model = FakeGLiNER(
            {"needs_task": {"task": 0.1, "no_task": 0.9}, "kind": {"chat": 1.0}, "response": {"talk": 1.0}}
        )

        decision = GLiNERRouter("unused", model=model).classify("Hello")

        self.assertFalse(decision.task)
        self.assertTrue(decision.talk)
        self.assertEqual(decision.kind, "chat")


class TalkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_lm_studio_api_uses_input_and_output_message(self):
        captured = {}

        def transport(url, body, headers):
            captured.update(url=url, body=json.loads(body))
            return json.dumps({"output": [{"type": "message", "content": "Blue."}]}).encode()

        talker = LMStudioTalker("http://localhost:1234/api/v1", "local-model", transport=transport)
        result = await talker.reply(TalkRequest("What color?", instructions=("Answer briefly.",)))

        self.assertEqual(result, "Blue.")
        self.assertEqual(captured["url"], "http://localhost:1234/api/v1/chat")
        self.assertEqual(captured["body"], {
            "model": "local-model",
            "input": "What color?",
            "store": False,
            "system_prompt": "Answer briefly.",
        })

    async def test_sends_context_and_returns_plain_reply(self):
        captured = {}

        def transport(url, body, headers):
            captured.update(url=url, body=json.loads(body), headers=headers)
            return json.dumps({"choices": [{"message": {"content": "The order shipped."}}]}).encode()

        talker = LMStudioTalker("http://localhost:1234/v1", "local-model", transport=transport)
        result = await talker.reply(
            TalkRequest(
                transcript="Where is my order?",
                instructions=("Be concise.",),
                thinking=("The user prefers short answers.",),
                history=(TranscriptTurn("user", "Hi"), TranscriptTurn("assistant", "Hello.")),
            )
        )

        self.assertEqual(result, "The order shipped.")
        self.assertEqual(captured["url"], "http://localhost:1234/v1/chat/completions")
        self.assertEqual(captured["body"]["model"], "local-model")
        self.assertEqual([item["role"] for item in captured["body"]["messages"]], ["system", "system", "user", "assistant", "user"])
        self.assertEqual(captured["headers"]["Content-Type"], "application/json")

    async def test_rejects_invalid_response(self):
        talker = LMStudioTalker("http://localhost:1234/v1", "local-model", transport=lambda *_: b"{}")

        with self.assertRaisesRegex(RuntimeError, "invalid chat response"):
            await talker.reply(TalkRequest("Hello"))


if __name__ == "__main__":
    unittest.main()
