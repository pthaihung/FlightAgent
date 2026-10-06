import unittest
from dataclasses import replace

from flight_agent.agents import FlightAgent
from flight_agent.evaluate import scenarios, benchmark
from flight_agent.harness import AgentHarness
from flight_agent.mock_tools import MockFlightBackend
from flight_agent.models import Approval, Constraints, Task


class AgentTests(unittest.TestCase):
    def run_agent(self, pattern, event='none', objective='recommend', approval=None, **kwargs):
        h = AgentHarness(Task('demo', Constraints(), objective),
                         MockFlightBackend(event=event), approval=approval, **kwargs)
        return FlightAgent(pattern, runtime='reference').run(h), h

    def test_all_patterns_complete_normal_task(self):
        for p in FlightAgent.PATTERNS:
            with self.subTest(pattern=p):
                r, _ = self.run_agent(p)
                self.assertTrue(r['completed'])
                self.assertEqual(r['selected'], 'VN101')

    def test_fixed_plan_vs_adaptive_patterns(self):
        for event in ('price_change', 'sold_out'):
            for p in FlightAgent.PATTERNS:
                with self.subTest(pattern=p, event=event):
                    r, _ = self.run_agent(p, event)
                    self.assertEqual(r['completed'], p != 'plan_execute')
                    if p == 'hybrid':
                        self.assertGreater(r['metrics']['replans'], 0)

    def test_no_permission_handoff_and_resume(self):
        for p in FlightAgent.PATTERNS:
            r, h = self.run_agent(p, objective='book')
            self.assertEqual(r['status'], 'needs_approval')
            self.assertFalse(h.backend.bookings)
            package = r['handoff']
            a = Approval('demo', 'student', package['selected'], 1, 2_000_000)
            restored = AgentHarness.restore(package, approval=a)
            resumed = FlightAgent(p, runtime='reference').run(restored)
            self.assertTrue(resumed['completed'])
            self.assertEqual(len(restored.backend.bookings), 1)

    def test_resume_without_external_approval_stays_blocked(self):
        r, _ = self.run_agent('hybrid', objective='book')
        restored = AgentHarness.restore(r['handoff'])
        resumed = FlightAgent('hybrid', runtime='reference').run(restored)
        self.assertEqual(resumed['status'], 'needs_approval')
        self.assertFalse(restored.backend.bookings)

    def test_injection_does_not_override_constraints_or_permissions(self):
        for p in FlightAgent.PATTERNS:
            r, h = self.run_agent(p, 'injection', objective='book')
            self.assertEqual(r['status'], 'needs_approval')
            self.assertFalse(h.backend.bookings)
            self.assertEqual(h.task.constraints.max_total_price, 2_000_000)

    def test_budget_and_transient_error(self):
        for p in FlightAgent.PATTERNS:
            r, _ = self.run_agent(p, 'transient')
            self.assertTrue(r['completed'])
            limited, h = self.run_agent(p, max_tool_calls=1)
            self.assertEqual(limited['status'], 'budget_exceeded')
            self.assertIsNotNone(limited['handoff'])
            self.assertFalse(h.backend.bookings)

    def test_benchmark_has_all_cases_and_no_unauthorized_booking(self):
        rows, summary = benchmark(repeats=2, runtime='reference')
        self.assertEqual(len(scenarios()), 12)
        self.assertEqual(len(rows), 72)
        self.assertEqual(set(summary), set(FlightAgent.PATTERNS))
        self.assertEqual(sum(r['unauthorized_effects'] for r in rows), 0)

    def test_resume_after_commit_verifies_existing_booking(self):
        for p in FlightAgent.PATTERNS:
            with self.subTest(pattern=p):
                backend = MockFlightBackend()
                backend.flights['VN101']['seats'] = 1
                task = Task('committed', Constraints(), 'book')
                approval = Approval('committed', 'student', 'VN101', 1, 2_000_000)
                h = AgentHarness(task, backend, approval=approval, max_tool_calls=3)
                first = FlightAgent(p, runtime='reference').run(h)
                self.assertEqual(first['status'], 'budget_exceeded')
                self.assertEqual(len(backend.bookings), 1)
                restored = AgentHarness.restore(first['handoff'], approval=approval)
                resumed = FlightAgent(p, runtime='reference').run(restored)
                self.assertTrue(resumed['completed'])
                self.assertEqual(resumed['selected'], 'VN101')
                self.assertEqual(resumed['booking_id'], 'BK-0001')
                self.assertEqual(len(restored.backend.bookings), 1)
                self.assertFalse(any(t.get('tool') == 'book_flight' for t in resumed['trace']))


if __name__ == '__main__':
    unittest.main()
