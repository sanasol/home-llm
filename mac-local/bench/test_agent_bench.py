import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import agent_bench as bench


class FakeAgent(BaseHTTPRequestHandler):
    """Streams a scripted agent: list files, read one file twice, submit the expected answer."""
    calls = []

    def log_message(self, *args): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeAgent.calls.append(body)
        turn = sum(1 for m in body['messages'] if m['role'] == 'assistant')
        script = [('list_files', {'path': 'runs'}), ('read_file', {'path': 'runs/2026-10-03.csv'}),
                  ('read_file', {'path': 'runs/2026-10-03.csv'}), ('read_file', {'path': '../../agent_bench.py'}),
                  ('submit_answer', {'answer': bench.expected_answer()})]
        name, args = script[turn]
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        chunks = [{'choices': [{'index': 0, 'delta': {'reasoning_content': 'thinking'}}]},
                  {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': f'c{turn}', 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(args)[:10]}}]}}]},
                  {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0,
                      'function': {'arguments': json.dumps(args)[10:]}}]}, 'finish_reason': 'tool_calls'}]},
                  {'choices': [], 'usage': {'prompt_tokens': 100 * (turn + 1), 'completion_tokens': 20,
                      'prompt_tokens_details': {'cached_tokens': 50 * turn},
                      'completion_tokens_details': {'reasoning_tokens': 5}},
                   'tensorfold': {'acceptance_rate': .8, 'prefill_seconds': .1}}]
        for chunk in chunks:
            self.wfile.write(b'data: ' + json.dumps(chunk).encode() + b'\n\n')
        self.wfile.write(b'data: [DONE]\n\n')


class BenchTests(unittest.TestCase):
    def test_fixture_answer_respects_traps(self):
        names = [x['job_name'] for x in bench.expected_answer()]
        self.assertEqual(names, ['etl-orders', 'index-rebuild', 'payout-batch'])
        payout = next(x for x in bench.expected_answer() if x['job_name'] == 'payout-batch')
        self.assertEqual(payout['failed_nights'], 3)  # the same-night retry counts once

    def test_grade(self):
        expected = bench.expected_answer()
        self.assertTrue(bench.grade(list(reversed(expected)), expected)['correct'])
        self.assertTrue(bench.grade(json.dumps(expected), expected)['correct'])
        wrong = [dict(x) for x in expected]
        wrong[0]['failed_duration_s'] += 1
        verdict = bench.grade(wrong, expected)
        self.assertFalse(verdict['correct'])
        self.assertEqual(verdict['matched'], 2)
        self.assertFalse(bench.grade(expected + [{'job_name': 'x'}], expected)['correct'])

    def test_tools_stay_inside_workspace(self):
        self.assertIn('2026-10-01.csv', bench.run_tool('list_files', {'path': 'runs'}))
        self.assertTrue(bench.run_tool('read_file', {'path': 'jobs.csv', 'max_lines': 2}).startswith('1: job_name'))
        with self.assertRaises(ValueError):
            bench.run_tool('read_file', {'path': '../agent_bench.py'})

    def test_agent_loop_against_streaming_server(self):
        http = ThreadingHTTPServer(('127.0.0.1', 0), FakeAgent)
        threading.Thread(target=http.serve_forever, daemon=True).start()
        args = SimpleNamespace(max_steps=10, max_tokens=64, time_limit=30, temperature=.6, top_p=.95)
        try:
            with patch.dict(bench.ENGINES, {'fake': dict(port=http.server_port, model='fake')}):
                result = bench.agent_run('fake', args, bench.expected_answer(), seed=1)
        finally:
            http.shutdown(); http.server_close()
        self.assertTrue(result['verdict']['correct'])
        self.assertEqual(result['steps'], 5)
        self.assertEqual(result['totals']['tool_calls'], 5)
        self.assertEqual(result['totals']['repeated_tool_calls'], 1)
        self.assertEqual(result['totals']['invalid_tool_calls'], 1)
        self.assertEqual(result['totals']['cached'], 500)
        self.assertEqual(result['step_log'][0]['engine_runtime']['acceptance_rate'], .8)
        self.assertIsNotNone(result['step_log'][0]['ttft'])
        last = FakeAgent.calls[-1]['messages']
        self.assertEqual(last[-1]['role'], 'tool')
        self.assertEqual(last[-1]['tool_call_id'], 'c3')

    def test_footprint_is_unknown_off_macos(self):
        with patch.object(bench.platform, 'system', return_value='Linux'):
            self.assertEqual(bench.footprint(1), (None, None))


if __name__ == '__main__':
    unittest.main()
