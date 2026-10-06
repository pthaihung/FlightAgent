import unittest
from dataclasses import replace

from flight_agent.models import Approval, Constraints, Task
from flight_agent.mock_tools import MockFlightBackend
from flight_agent.harness import AgentHarness


class HarnessTests(unittest.TestCase):
    def make(self, **kwargs):
        task = Task('t1', Constraints(), objective='book')
        return AgentHarness(task, MockFlightBackend(), **kwargs)

    def quote(self, h):
        return h.call('quote_flight', {'flight_id': 'VN101'})['data']

    def test_permission_blocks_side_effect(self):
        h = self.make()
        q = self.quote(h)
        r = h.call('book_flight', {'quote_id': q['quote_id']})
        self.assertEqual(r['error'], 'needs_approval')
        self.assertEqual(h.backend.bookings, {})

    def test_approval_is_bound_to_principal_task_and_flight(self):
        for changes in ({'principal': 'other'}, {'task_id': 'other'},
                        {'flight_id': 'VN202'}, {'max_total_price': 1}):
            with self.subTest(changes=changes):
                a = replace(Approval('t1', 'student', 'VN101', 1, 2_000_000), **changes)
                h = self.make(approval=a)
                q = self.quote(h)
                self.assertEqual(h.call('book_flight', {'quote_id': q['quote_id']})['error'],
                                 'needs_approval')
                self.assertFalse(h.backend.bookings)

    def test_completion_does_not_trust_agent_claim(self):
        h = self.make()
        r = h.finish('completed', selected='VN101', booking_id='invented')
        self.assertEqual(r['status'], 'failed')
        self.assertFalse(r['completed'])

    def test_booking_is_idempotent(self):
        h = self.make(approval=Approval('t1', 'student', 'VN101', 1, 2_000_000))
        q = self.quote(h)
        first = h.call('book_flight', {'quote_id': q['quote_id']})
        second = h.call('book_flight', {'quote_id': q['quote_id']})
        self.assertTrue(first['ok'])
        self.assertEqual(first['data']['booking_id'], second['data']['booking_id'])
        self.assertEqual(len(h.backend.bookings), 1)
        self.assertTrue(h.finish('completed', 'VN101', first['data']['booking_id'])['completed'])

    def test_constraints_enforced_before_booking(self):
        h = self.make(approval=Approval('t1', 'student', 'VN101', 1, 2_000_000))
        h.task = replace(h.task, constraints=replace(h.task.constraints, min_baggage_kg=20))
        q = self.quote(h)
        r = h.call('book_flight', {'quote_id': q['quote_id']})
        self.assertEqual(r['error'], 'constraint_violation')
        self.assertFalse(h.backend.bookings)

    def test_unknown_tool_invalid_args_and_budget(self):
        h = self.make(max_tool_calls=2)
        self.assertEqual(h.call('delete_all', {})['error'], 'tool_denied')
        self.assertEqual(h.call('quote_flight', {'flight_id': 12})['error'], 'invalid_arguments')
        self.assertEqual(h.call('search_flights', {})['error'], 'budget_exceeded')
        self.assertFalse(h.backend.bookings)

    def test_stale_quote_cannot_book(self):
        h = self.make(approval=Approval('t1', 'student', 'VN101', 1, 2_000_000))
        q = self.quote(h)
        h.backend.flights['VN101']['price'] += 100_000
        h.backend.flights['VN101']['version'] += 1
        self.assertEqual(h.call('book_flight', {'quote_id': q['quote_id']})['error'], 'stale_quote')
        self.assertFalse(h.backend.bookings)

    def test_invalid_input(self):
        for kwargs in ({'passengers': 0}, {'max_total_price': -1},
                       {'departure_date': 'not-date'}, {'origin': 'SGN', 'destination': 'SGN'}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    Constraints(**kwargs)

    def test_infeasible_claim_checked(self):
        h = self.make()
        self.assertEqual(h.finish('infeasible')['status'], 'failed')

    def test_handoff_preserves_allowlist(self):
        h = self.make(allowed_tools=('search_flights', 'quote_flight'))
        package = h.finish('failed')['handoff']
        resumed = AgentHarness.restore(package,
            approval=Approval('t1', 'student', 'VN101', 1, 2_000_000))
        q = self.quote(resumed)
        self.assertEqual(resumed.call('book_flight', {'quote_id': q['quote_id']})['error'], 'tool_denied')
        self.assertFalse(resumed.backend.bookings)

    def test_idempotency_when_last_seat_already_consumed(self):
        h = self.make(approval=Approval('t1', 'student', 'VN101', 1, 2_000_000))
        h.backend.flights['VN101']['seats'] = 1
        q = self.quote(h)
        first = h.call('book_flight', {'quote_id': q['quote_id']})
        second = h.call('book_flight', {'quote_id': q['quote_id']})
        self.assertTrue(second['ok'])
        self.assertEqual(first['data']['booking_id'], second['data']['booking_id'])
        self.assertEqual(h.backend.flights['VN101']['seats'], 0)

    def test_time_budget_checked_at_completion(self):
        h = self.make()
        h.started -= h.max_seconds + 1
        self.assertEqual(h.finish('completed', selected='VN101')['status'], 'budget_exceeded')

    def test_idempotency_identifiers_cannot_collide(self):
        backend = MockFlightBackend()
        bookings = []
        for principal, task_id in (('alice:group', 'trip'), ('alice', 'group:trip')):
            h = AgentHarness(Task(task_id, Constraints(), 'book'), backend, principal=principal,
                             approval=Approval(task_id, principal, 'VN101', 1, 2_000_000))
            q = self.quote(h)
            booking = h.call('book_flight', {'quote_id': q['quote_id']})['data']
            self.assertEqual(booking['principal'], principal)
            self.assertEqual(booking['task_id'], task_id)
            bookings.append(booking['booking_id'])
        self.assertEqual(len(set(bookings)), 2)


if __name__ == '__main__':
    unittest.main()
