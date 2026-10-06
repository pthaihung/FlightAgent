"""Resumable real-API benchmark. Infrastructure errors are not agent scores."""
import json
import os
from pathlib import Path

from flight_agent.agents import FlightAgent
from flight_agent.evaluate import run_scenario, save_results, scenarios, summarize_results


def is_api_error(row):
    reason = row.get('result', {}).get('reason') or ''
    prefixes = ('TooManyRequestsResponseError:', 'RateLimitError:', 'APIConnectionError:',
                'APITimeoutError:', 'AuthenticationError:', 'UnauthorizedResponseError:',
                'PaymentRequiredResponseError:', 'ServiceUnavailableResponseError:')
    return reason.startswith(prefixes) or 'free-models-per-day' in reason


def resolve_model(model):
    if model != 'openrouter':
        return model
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=True)
    slug = os.environ.get('OPENROUTER_MODEL', '').strip()
    if not slug or '/' not in slug:
        raise ValueError('Set OPENROUTER_MODEL to provider/model in .env')
    return f'openrouter:{slug}'


def run_benchmark_session(output, model, repeats=1, runtime='langgraph', max_seconds=180,
                          resume=False, progress=None):
    if not model:
        raise ValueError('A real API model is required')
    if type(repeats) is not int or repeats < 1:
        raise ValueError('repeats must be positive')
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    model = resolve_model(model)
    jobs = [(case, seed, pattern) for case in scenarios() for seed in range(repeats)
            for pattern in FlightAgent.PATTERNS]
    expected_keys = {(case.name, seed, pattern) for case, seed, pattern in jobs}
    runs_path, errors_path = directory / 'runs.json', directory / 'api_errors.json'
    previous = []
    if runs_path.exists():
        if not resume:
            raise ValueError('Results already exist; use --resume or a new output directory')
        previous = json.loads(runs_path.read_text(encoding='utf-8'))
    errors = json.loads(errors_path.read_text(encoding='utf-8')) if errors_path.exists() else []
    completed = {}
    for row in previous:
        key = (row['case'], row['seed'], row['architecture'])
        if row.get('model') != model or row['decision_mode'] != 'llm':
            raise ValueError('Cannot combine results from a different model or decision mode')
        if model.startswith('openrouter:') and row.get('result', {}).get(
                'reasoning_requested') != {'effort': 'none'}:
            raise ValueError('Cannot resume results with different or unknown reasoning settings; '
                             'use a new output directory')
        if row.get('max_seconds', 180) != max_seconds:
            raise ValueError('Resume must use the same time budget')
        if runtime != 'auto' and row['runtime'] != runtime:
            raise ValueError('Resume must use the same runtime')
        if key not in expected_keys:
            raise ValueError('Resume must include all existing cases/seeds')
        if is_api_error(row):
            if row not in errors:
                errors.append(row)
        else:
            if key in completed:
                raise ValueError('Duplicate benchmark result')
            completed[key] = row

    def checkpoint(stopped_reason=None):
        rows = [completed[(case.name, seed, pattern)] for case, seed, pattern in jobs
                if (case.name, seed, pattern) in completed]
        manifest = save_results(rows, summarize_results(rows), directory)
        manifest.update(expected_runs=len(jobs), evaluated_runs=len(rows),
                        pending_runs=len(jobs) - len(rows), api_error_attempts=len(errors),
                        benchmark_complete=len(rows) == len(jobs), stopped_reason=stopped_reason,
                        requested_model=model, time_budgets_seconds=[max_seconds])
        if model.startswith('openrouter:'):
            manifest['reasoning_requested'] = {'effort': 'none'}
        for path, data in ((directory / 'manifest.json', manifest), (errors_path, errors)):
            temporary = path.with_name(path.name + '.tmp')
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(path)
        return manifest

    manifest = checkpoint()
    for case, seed, pattern in jobs:
        key = (case.name, seed, pattern)
        if key in completed:
            continue
        if progress:
            progress(f'[{len(completed) + 1}/{len(jobs)}] {case.name} / {pattern} / seed {seed}')
        row = run_scenario(case, pattern, seed, runtime, model, max_seconds)
        if is_api_error(row):
            errors.append(row)
            manifest = checkpoint(row['result'].get('reason'))
            if progress:
                progress(f'API blocked; saved {len(completed)}/{len(jobs)} evaluations. Resume later.')
            return manifest
        completed[key] = row
        manifest = checkpoint()
        if progress:
            progress(f'  {row["status"]}; saved {len(completed)}/{len(jobs)}')
    return manifest
