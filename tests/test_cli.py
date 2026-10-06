import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from uuid import uuid4


class CliTests(unittest.TestCase):
    def test_benchmark_on_windows_legacy_console(self):
        directory = Path('tests') / f'cli-output-{uuid4().hex}'
        directory.mkdir()
        try:
            env = dict(os.environ, PYTHONIOENCODING='cp1252')
            run = subprocess.run([sys.executable, 'main.py', 'benchmark',
                                  '--runtime', 'reference', '--repeats', '1',
                                  '--output', str(directory)], env=env, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr.decode('cp1252', errors='replace'))
            self.assertIn('Mẫu', run.stdout.decode('utf-8'))
            summary = json.loads((Path(directory) / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['react']['runs'], 12)
        finally:
            for name in ('runs.json', 'summary.json', 'manifest.json', 'runs.csv', 'comparison.md'):
                (directory / name).unlink(missing_ok=True)
            directory.rmdir()


if __name__ == '__main__':
    unittest.main()
