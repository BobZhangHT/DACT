import unittest

from simulation import _run_task, make_tasks


class DriverTests(unittest.TestCase):
    def test_grids(self):
        d = make_tasks("testing", "demo", "c")
        f = make_tasks("testing", "full", "c")
        self.assertEqual(len(f), 10 * len(d))

    def test_worker(self):
        task = make_tasks("testing", "demo", "c")[0]
        task["method"] = "proposed_gaussian"
        result = _run_task(task)
        self.assertEqual(result["status"], "ok", result.get("traceback"))
