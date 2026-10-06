import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flight_agent.evaluate import run_scenario, scenarios
from flight_agent.benchmark_session import run_benchmark_session


class BenchmarkResumeTests(unittest.TestCase):
    def fake_run(self, case, pattern, seed, runtime, model, max_seconds):
        row = run_scenario(case, pattern, seed, 'reference')
        row.update(model=model, decision_mode='llm', max_seconds=max_seconds, runtime=runtime)
        row['result']['decision_mode'] = 'llm'
        row['result']['model'] = model
        row['result']['reasoning_requested'] = {'effort': 'none'}
        row['result']['llm_usage'] = dict(model_calls=3, input_tokens=100,
                                        output_tokens=20, usage_available=True)
        return row

    def test_stops_at_quota_then_resumes_only_missing_runs(self):
        case = scenarios()[0]
        calls = []

        def interrupted(*args):
            calls.append(args[1])
            row = self.fake_run(*args)
            if args[1] == 'plan_execute':
                row.update(status='failed', completed=False, correct=False)
                row['result'].update(status='failed', completed=False,
                    reason='TooManyRequestsResponseError: Rate limit exceeded: free-models-per-day')
            return row

        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='benchmark-test-') as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(Path.cwd().resolve()))
            with patch('flight_agent.benchmark_session.scenarios', return_value=[case]), \
                 patch('flight_agent.benchmark_session.run_scenario', side_effect=interrupted):
                manifest = run_benchmark_session(directory, 'openrouter:vendor/model', repeats=1)
            self.assertEqual(calls, ['react', 'plan_execute'])
            self.assertFalse(manifest['benchmark_complete'])
            self.assertEqual(manifest['evaluated_runs'], 1)
            self.assertEqual(manifest['pending_runs'], 2)
            summary = json.loads((Path(directory) / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['react']['handling_rate'], 1)
            self.assertNotIn('plan_execute', summary)
            calls.clear()

            def resumed(*args):
                calls.append(args[1])
                return self.fake_run(*args)

            with patch('flight_agent.benchmark_session.scenarios', return_value=[case]), \
                 patch('flight_agent.benchmark_session.run_scenario', side_effect=resumed):
                manifest = run_benchmark_session(directory, 'openrouter:vendor/model', repeats=1, resume=True)
            self.assertEqual(calls, ['plan_execute', 'hybrid'])
            self.assertTrue(manifest['benchmark_complete'])
            self.assertEqual(manifest['pending_runs'], 0)
            self.assertEqual(manifest['api_error_attempts'], 1)

    def test_cannot_mix_models_or_overwrite_results(self):
        case = scenarios()[0]
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='benchmark-test-') as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(Path.cwd().resolve()))
            with patch('flight_agent.benchmark_session.scenarios', return_value=[case]), \
                 patch('flight_agent.benchmark_session.run_scenario', side_effect=self.fake_run):
                run_benchmark_session(directory, 'openrouter:vendor/model', repeats=1)
                with self.assertRaisesRegex(ValueError, 'resume'):
                    run_benchmark_session(directory, 'openrouter:vendor/model', repeats=1)
                with self.assertRaisesRegex(ValueError, 'model'):
                    run_benchmark_session(directory, 'openrouter:vendor/other', repeats=1, resume=True)
                path = Path(directory) / 'runs.json'
                rows = json.loads(path.read_text(encoding='utf-8'))
                rows[0]['result'].pop('reasoning_requested')
                path.write_text(json.dumps(rows), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'reasoning'):
                    run_benchmark_session(directory, 'openrouter:vendor/model', repeats=1, resume=True)
