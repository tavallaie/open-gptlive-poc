import multiprocessing
import tempfile
import time
import unittest
from pathlib import Path

from open_gptlive_poc.adapters.sqlite_delegations import SQLiteDelegations


def _post_from_process(database: str, result_queue) -> None:
    caller = SQLiteDelegations(database)
    caller.initialize()
    callback_id = caller.enqueue("item_process", "From another process")
    result_queue.put(callback_id)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = caller.result(callback_id) if callback_id is not None else False
        if result is not None:
            result_queue.put(result)
            caller.close_worker()
            return
        time.sleep(0.02)
    result_queue.put(False)
    caller.close_worker()


class SQLiteDelegationTests(unittest.TestCase):
    def test_callback_crosses_independent_worker_stores_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "delegations.sqlite3")
            session_worker = SQLiteDelegations(database)
            callback_worker = SQLiteDelegations(database)
            session_worker.initialize()
            callback_worker.initialize()
            session_worker.register("item_1", "session_1")

            callback_id = callback_worker.enqueue("item_1", "Task finished")

            self.assertIsNotNone(callback_id)
            self.assertEqual(
                session_worker.pending(),
                [(callback_id, "item_1", "Task finished")],
            )
            self.assertIsNone(callback_worker.enqueue("item_1", "Duplicate"))
            session_worker.complete(callback_id, accepted=True)

            self.assertTrue(callback_worker.result(callback_id))
            self.assertIsNone(callback_worker.enqueue("item_1", "Late result"))
            session_worker.close_worker()
            callback_worker.close_worker()

    def test_closed_session_rejects_queued_callback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "delegations.sqlite3")
            owner = SQLiteDelegations(database)
            caller = SQLiteDelegations(database)
            owner.initialize()
            caller.initialize()
            owner.register("item_1", "session_1")
            callback_id = caller.enqueue("item_1", "Task finished")

            owner.unregister_session("session_1")

            self.assertFalse(caller.result(callback_id))
            self.assertEqual(owner.pending(), [])
            owner.close_worker()
            caller.close_worker()

    def test_callback_can_be_acknowledged_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "delegations.sqlite3")
            owner = SQLiteDelegations(database)
            owner.initialize()
            owner.register("item_process", "session_1")
            context = multiprocessing.get_context("spawn")
            result_queue = context.Queue()
            process = context.Process(target=_post_from_process, args=(database, result_queue))
            process.start()
            callback_id = result_queue.get(timeout=5)

            self.assertIsNotNone(callback_id)
            deadline = time.monotonic() + 5
            pending = []
            while not pending and time.monotonic() < deadline:
                pending = owner.pending()
                time.sleep(0.02)
            self.assertEqual(pending, [(callback_id, "item_process", "From another process")])
            owner.complete(callback_id, accepted=True)
            self.assertTrue(result_queue.get(timeout=5))
            process.join(timeout=5)
            self.assertEqual(process.exitcode, 0)
            owner.close_worker()


if __name__ == "__main__":
    unittest.main()
