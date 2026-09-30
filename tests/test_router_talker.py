import json
import asyncio
import tempfile
import unittest
from pathlib import Path

from open_gptlive_poc.adapters.gliner_router import GLiNERRouter
from open_gptlive_poc.adapters.http_tasks import HttpTasks
from open_gptlive_poc.adapters.laya_router import LayaRouter
from open_gptlive_poc.adapters.lmstudio_talker import LMStudioTalker
from open_gptlive_poc.adapters.sqlite_delegations import SQLiteDelegations
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
                "interrupt": {"interrupt_current_response": 0.9, "continue_current_response": 0.1},
                "turn_state": {"ready_to_process": 0.95, "wait_for_user": 0.05},
            }
        )
        router = GLiNERRouter("unused", model=model, task_threshold=0.75)

        decision = router.classify("Where is my order?")

        self.assertTrue(decision.task)
        self.assertTrue(decision.talk)
        self.assertEqual(decision.kind, "task")
        self.assertEqual(decision.labels["needs_task"], {"task": 0.8, "no_task": 0.2})
        self.assertTrue(decision.interrupt_current)
        self.assertFalse(decision.wait_for_user)
        self.assertEqual(model.calls[0][0], "Where is my order?")

    def test_chat_decision_does_not_schedule_task(self):
        model = FakeGLiNER(
            {"needs_task": {"task": 0.1, "no_task": 0.9}, "kind": {"chat": 1.0}, "response": {"talk": 1.0}}
        )

        decision = GLiNERRouter("unused", model=model).classify("Hello")

        self.assertFalse(decision.task)
        self.assertTrue(decision.talk)
        self.assertEqual(decision.kind, "chat")

    def test_wait_label_defers_task_and_response(self):
        model = FakeGLiNER({
            "needs_task": {"task": 0.0, "no_task": 1.0},
            "kind": {"chat": 1.0},
            "response": {"talk": 1.0},
            "turn_state": {"wait_for_user": 0.9, "ready_to_process": 0.1},
        })

        decision = GLiNERRouter("unused", model=model).classify("Hmm, I think we should…")

        self.assertTrue(decision.wait_for_user)
        self.assertFalse(decision.task)
        self.assertTrue(decision.talk)

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
    async def test_openai_stream_executes_tool_and_continues_reply(self):
        from open_gptlive_poc.live.session_tools import SessionTools

        calls = []

        async def stream_transport(url, payload):
            calls.append(payload.copy())
            if len(calls) == 1:
                yield "", {"choices": [{"delta": {"tool_calls": [{
                    "index": 0, "id": "call-time", "type": "function",
                    "function": {"name": "get_current_time", "arguments": "{}"},
                }]}}]}
            else:
                self.assertEqual(payload["messages"][-1]["role"], "tool")
                self.assertRegex(payload["messages"][-1]["content"], r"^\d{4}-\d\d-\d\dT.*[+-]\d\d:\d\d$")
                yield "", {"choices": [{"delta": {"content": "It is now 10:30."}}]}

        talker = LMStudioTalker("http://localhost:1234/v1", "local-model", stream_transport=stream_transport)
        tools = SessionTools(lambda _: True)
        result = [part async for part in talker.stream_reply(TalkRequest("What time is it?", tools=tools))]

        self.assertEqual(result, ["It is now 10:30."])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["tools"][0]["function"]["name"], "get_current_time")
        await tools.close()

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


class SessionToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_time_returns_local_iso_timestamp(self):
        from open_gptlive_poc.live.session_tools import SessionTools

        tools = SessionTools(lambda _: True)
        result = await tools.execute("get_current_time", {})
        self.assertRegex(result, r"^\d{4}-\d\d-\d\dT.*[+-]\d\d:\d\d$")
        await tools.close()

    async def test_timer_notifies_when_background_delay_finishes(self):
        from open_gptlive_poc.live.session_tools import SessionTools

        notified = asyncio.Event()
        messages = []

        def notify(message):
            messages.append(message)
            notified.set()
            return True

        tools = SessionTools(notify)
        result = await tools.execute("start_timer", {"seconds": 1, "message": "stretch"})
        self.assertIn("Timer started for 1 seconds", result)
        await asyncio.wait_for(notified.wait(), timeout=2)
        self.assertEqual(messages, ["Timer finished: stretch"])
        await tools.close()


class TasksTests(unittest.IsolatedAsyncioTestCase):
    async def test_posts_transcript_and_labels(self):
        captured = {}
        fast_callback = {}

        def transport(url, body, headers):
            captured.update(url=url, body=json.loads(body), headers=headers)
            fast_callback["id"] = delegations.enqueue("item_test", "Fast result")
            return b"{}"

        with tempfile.TemporaryDirectory() as directory:
            delegations = SQLiteDelegations(str(Path(directory) / "delegations.sqlite3"))
            delegations.initialize()
            delegation_id = "item_test"
            await HttpTasks(
                "http://tasks.local/delegations", delegations=delegations, transport=transport
            ).create(delegation_id, "sess_1", "Do it", {"kind": "task", "needs_task": 0.9})

        self.assertEqual(delegation_id, "item_test")
        self.assertEqual(captured["url"], "http://tasks.local/delegations")
        self.assertEqual(captured["body"]["delegation_id"], delegation_id)
        self.assertIsNotNone(fast_callback["id"])
        self.assertEqual(captured["body"]["session_id"], "sess_1")
        self.assertEqual(captured["body"]["transcript"], "Do it")
        self.assertEqual(captured["body"]["gliner"]["needs_task"], 0.9)

    async def test_http_failure_unregisters_delegation(self):
        with tempfile.TemporaryDirectory() as directory:
            delegations = SQLiteDelegations(str(Path(directory) / "delegations.sqlite3"))
            delegations.initialize()

            def fail_transport(*_):
                raise OSError("task service unavailable")

            with self.assertRaisesRegex(OSError, "unavailable"):
                await HttpTasks(
                    "http://tasks.local/delegations",
                    delegations=delegations,
                    transport=fail_transport,
                ).create("item_failed", "sess_1", "Do it", {"kind": "task"})

            self.assertIsNone(delegations.enqueue("item_failed", "late callback"))
            delegations.close_worker()


if __name__ == "__main__":
    unittest.main()
