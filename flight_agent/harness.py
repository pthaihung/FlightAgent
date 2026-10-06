"""Four harness layers, independent of agent reasoning and prompts.

Trust boundary: caller owns task, backend and approvals. Models receive only tool
schemas/results. Checkpoint files are trusted local artifacts, not model input.
"""
from copy import deepcopy
from time import perf_counter

from flight_agent.models import Task, ToolError, error, ok
from flight_agent.mock_tools import MockFlightBackend


class ConstraintPolicy:
    @staticmethod
    def check(flight, constraints, total_price=None, check_seats=True):
        c, f = constraints, flight
        total = f['price'] * c.passengers if total_price is None else total_price
        checks = {
            'origin': f['origin'] == c.origin,
            'destination': f['destination'] == c.destination,
            'date': f['departure_date'] == c.departure_date,
            'budget': total <= c.max_total_price,
            'stops': f['stops'] <= c.max_stops,
            'baggage': f['baggage_kg'] >= c.min_baggage_kg,
            'airline': not c.allowed_airlines or f['airline'] in c.allowed_airlines,
            'seats': not check_seats or f['seats'] >= c.passengers,
        }
        return checks

    @classmethod
    def eligible(cls, flight, constraints):
        return all(cls.check(flight, constraints).values())

    @classmethod
    def choose(cls, flights, constraints):
        eligible = [f for f in flights if cls.eligible(f, constraints)]
        return min(eligible, key=lambda f: (f['price'], f['flight_id'])) if eligible else None


class PermissionPolicy:
    def __init__(self, principal='student', approval=None, allowed_tools=None):
        self.principal, self.approval = principal, approval
        self.allowed_tools = frozenset(allowed_tools if allowed_tools is not None else
                                      ('search_flights', 'quote_flight', 'book_flight', 'get_booking'))

    def authorize_booking(self, task, quote):
        a = self.approval
        return bool(a and a.task_id == task.task_id and a.principal == self.principal
                    and a.flight_id == quote['flight_id']
                    and a.passengers == quote['passengers'] == task.constraints.passengers
                    and quote['total_price'] <= a.max_total_price
                    and quote['total_price'] <= task.constraints.max_total_price)


class CompletionVerifier:
    def verify(self, harness, status, selected=None, booking_id=None):
        backend, task = harness.backend, harness.task
        checks = {}
        evidence = {}
        if status == 'infeasible':
            candidates = [f for f in backend.flights.values()
                          if ConstraintPolicy.eligible(f, task.constraints)]
            checks['no_feasible_flight'] = not candidates
            return checks, evidence
        if status != 'completed':
            return checks, evidence
        flight = backend.flights.get(selected)
        checks['flight_exists'] = flight is not None
        if task.objective == 'book':
            booking = backend.bookings.get(booking_id)
            checks['booking_exists'] = booking is not None
            if booking:
                evidence['booking'] = deepcopy(booking)
                checks.update(
                    booking_confirmed=booking['status'] == 'confirmed',
                    booking_task=booking['task_id'] == task.task_id,
                    booking_principal=booking['principal'] == harness.permissions.principal,
                    booking_flight=booking['flight_id'] == selected,
                    booking_passengers=booking['passengers'] == task.constraints.passengers,
                    booking_authorized=harness.permissions.authorize_booking(task, booking),
                )
                checks.update(ConstraintPolicy.check(booking['flight_snapshot'], task.constraints,
                                                    booking['total_price'], check_seats=False))
        elif flight:
            evidence['flight'] = deepcopy(flight)
            checks.update(ConstraintPolicy.check(flight, task.constraints))
            cheapest = ConstraintPolicy.choose(list(backend.flights.values()), task.constraints)
            checks['cheapest_valid_flight'] = bool(cheapest and flight['price'] == cheapest['price'])
        return checks, evidence


class HandoffManager:
    @staticmethod
    def package(harness, reason, selected, plan):
        return {
            'schema_version': 1,
            'reason': reason,
            'task': harness.task.to_dict(),
            'principal': harness.permissions.principal,
            'allowed_tools': sorted(harness.permissions.allowed_tools),
            'selected': selected,
            'plan': deepcopy(plan),
            'completed_steps': deepcopy(harness.completed_steps),
            'trace': deepcopy(harness.trace),
            'metrics': deepcopy(harness.metrics),
            'pending_action': 'fresh_quote_and_external_approval' if reason == 'needs_approval'
                              else 'review_error_and_resume_with_new_budget',
            'backend_snapshot': harness.backend.snapshot(),
            # Approval deliberately absent: it must be supplied by trusted caller.
        }


class AgentHarness:
    SPECS = {'search_flights': {}, 'quote_flight': {'flight_id': str},
             'book_flight': {'quote_id': str}, 'get_booking': {'booking_id': str}}

    def __init__(self, task, backend, approval=None, principal='student',
                 allowed_tools=None, max_tool_calls=24, max_steps=32,
                 max_seconds=15, max_retries=1, max_replans=2):
        self.task, self.backend = task, backend
        self.permissions = PermissionPolicy(principal, approval, allowed_tools)
        self.verifier, self.handoffs = CompletionVerifier(), HandoffManager()
        for name, value in (('max_tool_calls', max_tool_calls), ('max_steps', max_steps),
                            ('max_retries', max_retries), ('max_replans', max_replans)):
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if max_seconds <= 0:
            raise ValueError('max_seconds must be positive')
        self.max_tool_calls, self.max_steps = max_tool_calls, max_steps
        self.max_seconds, self.max_retries, self.max_replans = max_seconds, max_retries, max_replans
        self.started = perf_counter()
        self.trace, self.completed_steps = [], []
        self.is_resumed = False
        self.metrics = dict(tool_calls=0, tool_errors=0, permission_denials=0,
                            decisions=0, plans=0, replans=0, retries=0, steps=0)

    def record(self, kind, **data):
        self.trace.append(dict(index=len(self.trace), kind=kind, **deepcopy(data)))

    def tick(self, kind):
        if self.metrics['steps'] >= self.max_steps or perf_counter() - self.started > self.max_seconds:
            raise ToolError('budget_exceeded', 'Agent step/time budget exhausted')
        self.metrics['steps'] += 1
        if kind in ('decisions', 'plans', 'replans'):
            self.metrics[kind] += 1

    def call(self, name, args):
        if self.metrics['tool_calls'] >= self.max_tool_calls or perf_counter() - self.started > self.max_seconds:
            result = error('budget_exceeded', 'Tool/time budget exhausted')
            self.record('tool_blocked', tool=name, result=result)
            return result
        self.metrics['tool_calls'] += 1  # Includes denied/invalid calls.
        try:
            if name not in self.SPECS or name not in self.permissions.allowed_tools:
                raise ToolError('tool_denied', 'Tool is not on the caller allowlist')
            spec = self.SPECS[name]
            if (not isinstance(args, dict) or set(args) != set(spec)
                    or any(type(args[k]) is not t or not args[k] for k, t in spec.items())):
                raise ToolError('invalid_arguments', 'Arguments do not match the tool schema')
            if name == 'search_flights':
                data = self.backend.search(self.task.constraints)
            elif name == 'quote_flight':
                data = self.backend.quote(args['flight_id'], self.task.constraints.passengers)
            elif name == 'get_booking':
                data = self.backend.get_booking(args['booking_id'])
                if data['principal'] != self.permissions.principal or data['task_id'] != self.task.task_id:
                    raise ToolError('tool_denied', 'Booking belongs to another principal or task')
            else:
                q = self.backend.get_quote(args['quote_id'])
                f = self.backend.get_flight(q['flight_id'])
                existing = self.backend.idempotency.get(self.backend.booking_key(
                    self.task.task_id, self.permissions.principal))
                previous = self.backend.bookings.get(existing)
                # A replay verifies the committed booking, not remaining inventory.
                checked_flight = previous['flight_snapshot'] if previous else f
                checked_total = previous['total_price'] if previous else q['total_price']
                if q['passengers'] != self.task.constraints.passengers or not all(
                        ConstraintPolicy.check(checked_flight, self.task.constraints,
                                               checked_total, check_seats=previous is None).values()):
                    raise ToolError('constraint_violation', 'Booking does not satisfy task constraints')
                if self.task.objective != 'book' or not self.permissions.authorize_booking(self.task, q):
                    raise ToolError('needs_approval', 'Caller approval required for this exact booking')
                data = self.backend.book(args['quote_id'], self.task.task_id, self.permissions.principal)
            result = ok(data)
        except ToolError as exc:
            result = error(exc.code, str(exc))
            self.metrics['tool_errors'] += 1
            if exc.code in ('needs_approval', 'tool_denied'):
                self.metrics['permission_denials'] += 1
        self.record('tool', tool=name, args=args, result=result)
        return result

    def as_langchain_tools(self):
        """Real LangChain tools; each entry still goes through the harness."""
        from langchain_core.tools import StructuredTool

        def search_flights() -> dict:
            """Search flights for the immutable caller task."""
            return self.call('search_flights', {})

        def quote_flight(flight_id: str) -> dict:
            """Get a fresh total price and seat availability for a flight."""
            return self.call('quote_flight', {'flight_id': flight_id})

        def book_flight(quote_id: str) -> dict:
            """Book a quote only if caller constraints and external approval allow it."""
            return self.call('book_flight', {'quote_id': quote_id})

        def get_booking(booking_id: str) -> dict:
            """Retrieve this task's booking to verify its confirmed status."""
            return self.call('get_booking', {'booking_id': booking_id})

        return [StructuredTool.from_function(fn) for fn in
                (search_flights, quote_flight, book_flight, get_booking)
                if fn.__name__ in self.permissions.allowed_tools]

    def finish(self, status, selected=None, booking_id=None, plan=None, reason=None):
        if status in ('completed', 'infeasible') and perf_counter() - self.started > self.max_seconds:
            status, reason = 'budget_exceeded', 'Time budget exceeded before verification'
        checks, evidence = self.verifier.verify(self, status, selected, booking_id)
        if status in ('completed', 'infeasible') and (not checks or not all(checks.values())):
            status, reason = 'failed', 'Completion claim failed backend verification'
        self.record('verification', status=status, checks=checks, evidence=evidence)
        handoff = self.handoffs.package(self, status, selected, plan or []) if status in (
            'needs_approval', 'failed', 'budget_exceeded') else None
        return dict(task_id=self.task.task_id, status=status, completed=status == 'completed',
                    selected=selected, booking_id=booking_id, checks=checks, evidence=evidence,
                    reason=reason, handoff=handoff, trace=deepcopy(self.trace),
                    metrics=dict(self.metrics, elapsed_ms=(perf_counter() - self.started) * 1000))

    @classmethod
    def restore(cls, package, approval=None, **budgets):
        """Restore trusted local state, with fresh caller approval and fresh budgets.

        Not an authentication mechanism: production checkpoints require storage
        access control/signatures. Never accept this dict from an LLM.
        """
        if package.get('schema_version') != 1:
            raise ValueError('Unsupported checkpoint version')
        h = cls(Task.from_dict(package['task']), MockFlightBackend.restore(package['backend_snapshot']),
                approval=approval, principal=package['principal'],
                allowed_tools=package['allowed_tools'], **budgets)
        h.record('resume', previous_reason=package['reason'], previous_metrics=package['metrics'])
        h.is_resumed = True
        h.completed_steps = deepcopy(package['completed_steps'])
        return h
