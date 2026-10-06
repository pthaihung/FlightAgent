"""Run actual dependency integrations when installed; never fake the imports."""
from importlib.util import find_spec
import unittest

from flight_agent.agents import FlightAgent
from flight_agent.evaluate import scenarios, run_scenario
from flight_agent.harness import AgentHarness
from flight_agent.mock_tools import MockFlightBackend
from flight_agent.models import Constraints, Task


class IntegrationTests(unittest.TestCase):
    @unittest.skipUnless(find_spec('langgraph'), 'LangGraph not installed; network unavailable')
    def test_real_langgraph_matches_reference_statuses(self):
        for case in scenarios():
            for pattern in FlightAgent.PATTERNS:
                with self.subTest(case=case.name, pattern=pattern):
                    actual = run_scenario(case, pattern, 0, runtime='langgraph')
                    reference = run_scenario(case, pattern, 0, runtime='reference')
                    self.assertEqual(actual['status'], reference['status'])
                    self.assertEqual(actual['runtime'], 'langgraph')
                    self.assertEqual(actual['unauthorized_effects'], 0)

    @unittest.skipUnless(find_spec('langchain_core'), 'LangChain Core not installed; network unavailable')
    def test_real_structured_tools_cannot_bypass_permission(self):
        h = AgentHarness(Task('integration', Constraints(), 'book'), MockFlightBackend())
        tools = {t.name: t for t in h.as_langchain_tools()}
        quote = tools['quote_flight'].invoke({'flight_id': 'VN101'})['data']
        denied = tools['book_flight'].invoke({'quote_id': quote['quote_id']})
        self.assertEqual(denied['error'], 'needs_approval')
        self.assertFalse(h.backend.bookings)


if __name__ == '__main__':
    unittest.main()
