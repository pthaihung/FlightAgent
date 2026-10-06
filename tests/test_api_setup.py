import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from flight_agent.agents import MockDecisionModel
from flight_agent.evaluate import run_scenario, scenarios


class InstrumentedModel(MockDecisionModel):
    mode = 'llm'
    usage = {'model_calls': 1, 'input_tokens': 100, 'output_tokens': 20, 'usage_available': True}


class ApiSetupTests(unittest.TestCase):
    def test_action_schema_rejects_invented_search_arguments(self):
        from langchain_core.tools import StructuredTool
        from flight_agent.agents import LangChainDecisionModel
        from flight_agent.harness import AgentHarness
        from flight_agent.models import Task, Constraints
        from flight_agent.mock_tools import MockFlightBackend
        from pydantic import ValidationError
        chat = MagicMock()
        chat.with_structured_output.side_effect = lambda schema: schema
        with patch.dict('sys.modules', {'langchain_openrouter': SimpleNamespace(
                ChatOpenRouter=MagicMock(return_value=chat))}), \
             patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-only-key'}), \
             patch('dotenv.load_dotenv'):
            model = LangChainDecisionModel('openrouter:vendor/test')
        harness = AgentHarness(Task('schema-test', Constraints()), MockFlightBackend())
        state = dict(selected=None, quote_id=None, booking_id=None)
        def response(schema, payload):
            return schema.model_validate({'name': 'search_flights', 'args': {'origin': 'SGN'}})
        with patch.object(model, '_invoke', side_effect=response):
            with self.assertRaises(ValidationError):
                model.action('search', state, harness)
        for phase, name, args in [('search', 'search_flights', {}),
                                 ('quote', 'quote_flight', {'flight_id': 'VN101'}),
                                 ('book', 'book_flight', {'quote_id': 'Q1'}),
                                 ('lookup', 'get_booking', {'booking_id': 'B1'})]:
            with self.subTest(phase=phase), patch.object(model, '_invoke',
                    side_effect=lambda schema, payload: schema.model_validate(
                        {'name': name, 'args': args})):
                self.assertEqual(model.action(phase, state, harness), {'name': name, 'args': args})

    def test_openrouter_uses_its_own_key_endpoint_and_model(self):
        from flight_agent.agents import LangChainDecisionModel
        factory = MagicMock()
        with patch.dict('sys.modules', {'langchain_openrouter': SimpleNamespace(ChatOpenRouter=factory)}), \
             patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-only-key',
                                       'OPENROUTER_MODEL': 'vendor/test-model'}, clear=True), \
             patch('dotenv.load_dotenv'):
            model = LangChainDecisionModel('openrouter')
        self.assertEqual(factory.call_args.kwargs['model'], 'vendor/test-model')
        self.assertEqual(factory.call_args.kwargs['api_key'], 'test-only-key')
        self.assertEqual(factory.call_args.kwargs['openrouter_api_base'], 'https://openrouter.ai/api/v1')
        self.assertEqual(factory.call_args.kwargs['reasoning'], {'effort': 'none'})
        from openrouter import components
        from openrouter.utils import marshal_json
        import json
        # The installed SDK discards unknown `enabled`, so verify the wire shape.
        self.assertEqual(json.loads(marshal_json(factory.call_args.kwargs['reasoning'],
                         components.ChatRequestReasoning)), {'effort': 'none'})
        self.assertEqual(model.model_name, 'openrouter:vendor/test-model')

    def test_openrouter_missing_key_has_clear_error(self):
        from flight_agent.agents import LangChainDecisionModel
        with patch.dict('os.environ', {}, clear=True), patch('dotenv.load_dotenv'):
            with self.assertRaisesRegex(ValueError, 'OPENROUTER_API_KEY'):
                LangChainDecisionModel('openrouter:vendor/test-model')

    def test_llm_receives_larger_time_budget_and_model_metadata(self):
        from flight_agent.harness import AgentHarness
        received = []

        def capture(*args, **kwargs):
            h = AgentHarness(*args, **kwargs)
            received.append(h.max_seconds)
            return h

        with patch('flight_agent.evaluate.LangChainDecisionModel', return_value=InstrumentedModel()), \
             patch('flight_agent.evaluate.AgentHarness', side_effect=capture):
            row = run_scenario(scenarios()[0], 'react', 0, 'reference', 'google_genai:test')
        self.assertEqual(received, [180])
        self.assertEqual(row['model'], 'google_genai:test')
        self.assertTrue(row['completed'])

    def test_caller_can_set_time_budget(self):
        row = run_scenario(scenarios()[0], 'react', 0, 'reference', max_seconds=0.000001)
        self.assertEqual(row['status'], 'budget_exceeded')


if __name__ == '__main__':
    unittest.main()
