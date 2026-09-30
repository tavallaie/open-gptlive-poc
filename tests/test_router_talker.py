import json
import unittest

from open_gptlive_poc.adapters.gliner_router import GLiNERRouter
from open_gptlive_poc.adapters.http_tasks import HttpTasks
from open_gptlive_poc.adapters.laya_router import LayaRouter
from open_gptlive_poc.adapters.lmstudio_talker import LMStudioTalker
from open_gptlive_poc.ports.talker import TalkRequest, TranscriptTurn


class FakeGLiNER:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def classify_text(self, transcript, questions, include_confidence=False):
        self.calls.append((transcript, questions, include_confidence))
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

    def test_laya_decision_uses_typed_answers(self):
        class FakeLaya:
            def __init__(self):
                self.calls = []

            def predict(self, state, questions):
                self.calls.append((state, questions))
                return {"answers": {
                    "needs_task": {"choice": "task", "probabilities": {"task": 0.9, "no_task": 0.1}},
                    "kind": {"choice": "task"},
                    "response": {"choice": "both"},
                }}

        model = FakeLaya()
        decision = LayaRouter("unused", onnx_path="unused", model=model).classify("Book a flight")

        self.assertTrue(decision.task)
        self.assertTrue(decision.talk)
        self.assertEqual(decision.kind, "task")
        self.assertEqual(model.calls[0][0], "Book a flight")


class TalkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_stream_yields_message_deltas(self):
        async def stream_transport(url, payload):
            self.assertEqual(url, "http://localhost:1234/api/v1/chat")
            self.assertTrue(payload["stream"])
            yield "chat.start", {"type": "chat.start"}
            yield "message.delta", {"type": "message.delta", "content": "Blue"}
            yield "message.delta", {"type": "message.delta", "content": "."}
            yield "message.end", {"type": "message.end"}

        talker = LMStudioTalker("http://localhost:1234/api/v1", "local-model", stream_transport=stream_transport)

        self.assertEqual([fragment async for fragment in talker.stream_reply(TalkRequest("What color?"))], ["Blue", "."])

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
        self.assertEqual(captured["body"]["reasoning_effort"], "none")
        self.assertEqual([item["role"] for item in captured["body"]["messages"]], ["system", "system", "user", "assistant", "user"])
        self.assertEqual(captured["headers"]["Content-Type"], "application/json")

    async def test_rejects_invalid_response(self):
        talker = LMStudioTalker("http://localhost:1234/v1", "local-model", transport=lambda *_: b"{}")

        with self.assertRaisesRegex(RuntimeError, "invalid chat response"):
            await talker.reply(TalkRequest("Hello"))


class TasksTests(unittest.IsolatedAsyncioTestCase):
    async def test_posts_transcript_and_labels(self):
        captured = {}

        def transport(url, body, headers):
            captured.update(url=url, body=json.loads(body), headers=headers)
            return b"{}"

        delegation_id = await HttpTasks("http://tasks.local/delegations", transport=transport).create(
            "sess_1", "Do it", {"kind": "task", "needs_task": 0.9}
        )

        self.assertTrue(delegation_id.startswith("item_"))
        self.assertEqual(captured["url"], "http://tasks.local/delegations")
        self.assertEqual(captured["body"]["session_id"], "sess_1")
        self.assertEqual(captured["body"]["transcript"], "Do it")
        self.assertEqual(captured["body"]["gliner"]["needs_task"], 0.9)


if __name__ == "__main__":
    unittest.main()
