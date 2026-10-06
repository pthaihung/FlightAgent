"""Entry point: python main.py demo|benchmark|resume --help."""
import argparse
import json
import sys
from pathlib import Path

from flight_agent.agents import FlightAgent, LangChainDecisionModel
from flight_agent.evaluate import benchmark, comparison_table, run_scenario, save_results, scenarios
from flight_agent.harness import AgentHarness
from flight_agent.models import Approval
from flight_agent.benchmark_session import run_benchmark_session


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description='FlightAgent harness assignment')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('demo', 'benchmark', 'resume'):
        cmd = sub.add_parser(command)
        cmd.add_argument('--runtime', choices=['auto', 'reference', 'langgraph'], default='auto')
        cmd.add_argument('--model', help='Opt-in LLM model, e.g. provider:model; requires provider package/key')
        cmd.add_argument('--max-seconds', type=float, help='Per-task time budget; default 180 for LLM, 15 for mock')
        if command != 'benchmark':
            cmd.add_argument('--pattern', choices=FlightAgent.PATTERNS, default='hybrid')
        if command == 'demo':
            cmd.add_argument('--case', choices=[c.name for c in scenarios()], default='normal')
            cmd.add_argument('--seed', type=int, default=0)
            cmd.add_argument('--output', default='results/demo.json')
        elif command == 'benchmark':
            cmd.add_argument('--repeats', type=int, default=5)
            cmd.add_argument('--output', default='results')
            cmd.add_argument('--resume', action='store_true', help='Resume saved API evaluations; retry API errors only')
        else:
            cmd.add_argument('checkpoint', help='Trusted demo JSON containing handoff or handoff JSON')
            cmd.add_argument('--approval', help='Caller-created approval JSON file; never from agent output')
            cmd.add_argument('--output', default='results/resume.json')
    args = parser.parse_args()
    if args.command == 'benchmark':
        if args.model:
            manifest = run_benchmark_session(args.output, args.model, args.repeats, args.runtime,
                args.max_seconds if args.max_seconds is not None else 180, args.resume,
                progress=lambda message: print(message, flush=True))
            print(json.dumps(manifest, ensure_ascii=True, indent=2))
            if not manifest['benchmark_complete']:
                raise SystemExit(2)
            return
        if args.resume:
            parser.error('--resume requires --model')
        rows, summary = benchmark(args.repeats, args.runtime, args.model, args.max_seconds)
        manifest = save_results(rows, summary, args.output)
        print(json.dumps(manifest, ensure_ascii=True, indent=2))
        print(comparison_table(summary))
        return
    if args.command == 'demo':
        case = next(c for c in scenarios() if c.name == args.case)
        result = run_scenario(case, args.pattern, args.seed, args.runtime, args.model, args.max_seconds)['result']
    else:
        package = json.loads(Path(args.checkpoint).read_text(encoding='utf-8-sig'))
        package = package.get('handoff', package)
        approval = Approval(**json.loads(Path(args.approval).read_text(encoding='utf-8-sig'))) if args.approval else None
        harness = AgentHarness.restore(package, approval,
            max_seconds=args.max_seconds if args.max_seconds is not None else (180 if args.model else 15))
        result = FlightAgent(args.pattern, args.runtime,
                             LangChainDecisionModel(args.model) if args.model else None).run(harness)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    concise = {k: result[k] for k in ('architecture', 'runtime', 'decision_mode', 'status',
                                      'selected', 'booking_id', 'checks', 'metrics', 'reason')}
    print(json.dumps(concise, ensure_ascii=True, indent=2))
    print(f'Full trace/checkpoint: {destination}')


if __name__ == '__main__':
    main()
