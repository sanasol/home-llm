#!/usr/bin/env python3
"""Agentic A/B benchmark for the local engines: time to a *correct* answer, steps and memory.

Standard library only (Python 3.9+). Every run is saved under bench/results/ and RESULTS.md is
regenerated from all saved runs. Prompts are synthetic; no Hermes conversations are read.

    python3 mac-local/bench/agent_bench.py --engine tensorfold --manage --runs 3
    python3 mac-local/bench/agent_bench.py --engine sushi --manage --runs 3
"""
import argparse
import csv
import ctypes
import datetime
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / 'fixtures/nightly-history'
RESULTS = HERE / 'results'
DASHBOARD = 'http://127.0.0.1:12347'
MEMORY_TARGET_GIB = 20.0
ENGINES = {
    'sushi': dict(port=12345, model='Qwen3.8-Flash-Next-Abliterated-Sushi-2bpw'),
    'bonsai': dict(port=12346, model='Bonsai-2-27B'),
    'tensorfold': dict(port=12348, model='Qwen3.8-27B-DFlash2'),
}

# ---------------------------------------------------------------- fixture

JOBS = ['backup-db', 'billing-export', 'cdn-purge', 'crm-sync', 'dns-audit', 'email-digest', 'etl-orders',
        'etl-users', 'feature-flags', 'geoip-refresh', 'image-resize', 'index-rebuild', 'invoice-pdf',
        'ledger-close', 'log-rotate', 'metrics-rollup', 'ml-retrain', 'news-feed', 'payout-batch',
        'pricing-cache', 'report-weekly', 'search-warmup', 'sitemap-gen', 'sms-queue', 'stock-import',
        'tax-calc', 'thumbnail-gc', 'tls-renew', 'user-cleanup', 'vat-report']
TEAMS = ['core', 'data', 'finance', 'growth', 'infra', 'platform']
NIGHTS = [f'2026-10-0{d}' for d in range(1, 8)]
DEFAULT_COLUMNS = ['night', 'job_name', 'host', 'started_at', 'duration_s', 'status', 'exit_code', 'message']
REORDERED = ['job_name', 'status', 'duration_s', 'host', 'night', 'exit_code', 'message', 'started_at']
WORDS = ('connection pool drained retry scheduled upstream latency spike checksum verified shard rebalanced '
         'cache warm partial write lock contention quota exceeded snapshot uploaded manifest stale worker '
         'restarted batch committed cursor advanced backlog cleared token refreshed').split()
# Planted cases make the answer sensitive to column order, retries and timeout handling.
PLANTED = {
    'etl-orders': {'2026-10-02': ['failed'], '2026-10-03': ['failed'], '2026-10-06': ['failed']},
    'payout-batch': {'2026-10-01': ['failed'], '2026-10-05': ['failed', 'failed'], '2026-10-07': ['failed']},
    'ml-retrain': {'2026-10-02': ['failed'], '2026-10-04': ['timeout'], '2026-10-05': ['timeout'],
                   '2026-10-06': ['failed']},
    'index-rebuild': {'2026-10-01': ['failed'], '2026-10-03': ['failed'], '2026-10-04': ['failed'],
                      '2026-10-06': ['failed']},
    'crm-sync': {'2026-10-02': ['failed'], '2026-10-05': ['failed', 'ok']},
    'tls-renew': {'2026-10-03': ['failed'], '2026-10-06': ['failed'], '2026-10-07': ['timeout']},
}
QUESTION = (
    'The directory contains jobs.csv (job owners) and runs/, one CSV per night of scheduled job runs. '
    'Find every job whose status was exactly "failed" on at least 3 distinct nights. A retry on the same '
    'night counts as one night; "timeout" is not "failed". Column order is not the same in every file: '
    'read each header. For each such job report job_name, owner_team, failed_nights (number of distinct '
    'nights) and failed_duration_s (sum of duration_s over all its failed runs, retries included). '
    'Call submit_answer once with the list sorted by job_name.')


def make_fixture(path=FIXTURE):
    rng = random.Random(20261010)
    (path / 'runs').mkdir(parents=True, exist_ok=True)
    with (path / 'jobs.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['job_name', 'owner_team', 'schedule'])
        for job in JOBS:
            w.writerow([job, rng.choice(TEAMS), f'{rng.randint(0, 5):02d}:{rng.choice([0, 15, 30, 45]):02d}'])
    noise_failures = {}
    for night in NIGHTS:
        columns = REORDERED if night in ('2026-10-03', '2026-10-06') else DEFAULT_COLUMNS
        rows = []
        for job in JOBS:
            planted = PLANTED.get(job, {})
            if job in PLANTED:
                statuses = planted.get(night, ['ok'])
            else:
                # Background noise: at most two failures per unplanted job, never qualifying.
                status = rng.choices(['ok', 'failed', 'timeout', 'skipped'], [76, 9, 5, 10])[0]
                if status == 'failed':
                    noise_failures[job] = noise_failures.get(job, 0) + 1
                    if noise_failures[job] > 2: status = 'ok'
                statuses = [status]
            for attempt, status in enumerate(statuses):
                start = f'{night}T{rng.randint(0, 5):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z'
                rows.append(dict(night=night, job_name=job, host=f'worker-{rng.randint(1, 12):02d}',
                    started_at=start, duration_s=0 if status == 'skipped' else rng.randint(20, 3600),
                    status=status, exit_code={'ok': 0, 'skipped': 0, 'timeout': 124}.get(status, rng.choice([1, 2, 137])),
                    message=' '.join(rng.choice(WORDS) for _ in range(rng.randint(8, 14))) +
                            (f' (retry {attempt})' if attempt else '')))
        with (path / 'runs' / f'{night}.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, columns)
            w.writeheader()
            w.writerows(rows)
    noise_guard(path)


def noise_guard(path):
    for job, nights in failures(path).items():
        if job not in PLANTED and len(nights) >= 3:
            raise RuntimeError(f'Fixture noise qualifies unexpectedly: {job}')


def failures(path=FIXTURE):
    result = {}
    for file in sorted((path / 'runs').glob('*.csv')):
        with file.open(newline='') as f:
            for row in csv.DictReader(f):
                if row['status'] == 'failed':
                    result.setdefault(row['job_name'], {}).setdefault(row['night'], []).append(int(row['duration_s']))
    return result


def expected_answer(path=FIXTURE):
    with (path / 'jobs.csv').open(newline='') as f:
        owners = {r['job_name']: r['owner_team'] for r in csv.DictReader(f)}
    answer = []
    for job, nights in sorted(failures(path).items()):
        if len(nights) >= 3:
            answer.append(dict(job_name=job, owner_team=owners[job], failed_nights=len(nights),
                               failed_duration_s=sum(sum(v) for v in nights.values())))
    return answer


def grade(answer, expected):
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except ValueError:
            return dict(correct=False, matched=0, expected=len(expected), reason='answer is not JSON')
    if isinstance(answer, dict):
        answer = next((v for v in answer.values() if isinstance(v, list)), answer)
    if not isinstance(answer, list):
        return dict(correct=False, matched=0, expected=len(expected), reason='answer is not a list')
    keys = ('job_name', 'owner_team', 'failed_nights', 'failed_duration_s')
    def norm(x):
        try:
            return (str(x['job_name']).strip(), str(x['owner_team']).strip(), int(x['failed_nights']),
                    int(float(x['failed_duration_s'])))
        except (KeyError, TypeError, ValueError):
            return None
    got = [norm(x) if isinstance(x, dict) else None for x in answer]
    want = [tuple(e[k] for k in keys) for e in expected]
    matched = len(set(want) & set(g for g in got if g))
    return dict(correct=sorted(g for g in got if g) == sorted(want) and None not in got,
                matched=matched, expected=len(want), submitted=len(got))

# ---------------------------------------------------------------- tools

TOOLS = [
    {'type': 'function', 'function': {'name': 'list_files', 'description': 'List files under a directory of the workspace.',
     'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': 'Relative directory, "." for root'}},
                    'required': ['path']}}},
    {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a text file. Lines are numbered from 1.',
     'parameters': {'type': 'object', 'properties': {
         'path': {'type': 'string'}, 'start_line': {'type': 'integer', 'default': 1},
         'max_lines': {'type': 'integer', 'default': 200}}, 'required': ['path']}}},
    {'type': 'function', 'function': {'name': 'submit_answer', 'description': 'Submit the final answer once.',
     'parameters': {'type': 'object', 'properties': {'answer': {'type': 'array', 'items': {'type': 'object', 'properties': {
         'job_name': {'type': 'string'}, 'owner_team': {'type': 'string'}, 'failed_nights': {'type': 'integer'},
         'failed_duration_s': {'type': 'integer'}}}}}, 'required': ['answer']}}},
]


def run_tool(name, args, root=FIXTURE):
    def inside(rel):
        path = (root / (rel or '.')).resolve()
        if path != root.resolve() and not str(path).startswith(str(root.resolve()) + os.sep):
            raise ValueError('path outside workspace')
        return path
    if name == 'list_files':
        path = inside(args.get('path', '.'))
        return '\n'.join(sorted(str(p.relative_to(root)) + ('/' if p.is_dir() else '') for p in path.iterdir()))
    if name == 'read_file':
        lines = inside(args['path']).read_text().splitlines()
        start = max(1, int(args.get('start_line') or 1))
        count = max(1, min(400, int(args.get('max_lines') or 200)))
        chunk = lines[start - 1:start - 1 + count]
        tail = f'\n[lines {start}-{start + len(chunk) - 1} of {len(lines)}]'
        return '\n'.join(f'{start + i}: {l}' for i, l in enumerate(chunk)) + tail
    raise ValueError(f'unknown tool {name}')

# ---------------------------------------------------------------- memory

class RUsage(ctypes.Structure):
    # rusage_info_v4: phys_footprint at offset 72, lifetime_max_phys_footprint at 240.
    _fields_ = [('uuid', ctypes.c_uint8 * 16), ('v', ctypes.c_uint64 * 40)]


def footprint(pid):
    """(current, lifetime peak) physical footprint in bytes, including Metal buffers; macOS only."""
    if not pid or platform.system() != 'Darwin':
        return None, None
    try:
        lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        info = RUsage()
        if lib.proc_pid_rusage(int(pid), 4, ctypes.byref(info)) != 0:
            return None, None
        return info.v[(72 - 16) // 8], info.v[(240 - 16) // 8]
    except OSError:
        return None, None


def swap_used_mib():
    try:
        out = subprocess.check_output(['sysctl', '-n', 'vm.swapusage'], text=True)
        return float(re.search(r'used = ([\d.]+)M', out).group(1))
    except (OSError, subprocess.CalledProcessError, AttributeError):
        return None


def memory_level():
    try:
        return int(subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_level'], text=True))
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def server_pid(engine):
    try:
        pid = json.load(urlopen(Request(f'{DASHBOARD}/api/status?engine={engine}'), timeout=3)).get('pid')
        if pid: return pid
    except (OSError, ValueError):
        pass
    try:
        out = subprocess.check_output(['lsof', '-ti', f'tcp:{ENGINES[engine]["port"]}', '-sTCP:LISTEN'], text=True)
        return int(out.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def engine_memory(engine):
    """The engine's own memory reading in bytes, where it publishes one."""
    if engine not in ENGINES:
        return None
    port = ENGINES[engine]['port']
    try:
        if engine == 'sushi':
            g = json.load(urlopen(f'http://127.0.0.1:{port}/metrics.json', timeout=3)).get('gauges', {})
            return g.get('mlx_active_bytes', 0) + g.get('mlx_cache_bytes', 0)
        if engine == 'tensorfold':
            text = urlopen(f'http://127.0.0.1:{port}/metrics', timeout=3).read().decode()
            m = re.search(r'^tensorfold:process_footprint_bytes (\d+)', text, re.M)
            return int(m.group(1)) if m else None
    except (OSError, ValueError):
        return None
    return None


class Sampler(threading.Thread):
    def __init__(self, engine, pid, interval=.5):
        super().__init__(daemon=True)
        self.engine, self.pid, self.interval = engine, pid, interval
        self.samples, self.stop = [], threading.Event()

    def run(self):
        tick = 0
        while not self.stop.is_set():
            current, peak = footprint(self.pid)
            sample = dict(t=time.time(), footprint=current, lifetime_peak=peak)
            if tick % 4 == 0:
                sample['engine_bytes'] = engine_memory(self.engine)
            self.samples.append(sample)
            tick += 1
            self.stop.wait(self.interval)

    def summary(self):
        values = [s['footprint'] for s in self.samples if s['footprint'] is not None]
        lifetime = [s['lifetime_peak'] for s in self.samples if s['lifetime_peak'] is not None]
        engine = [s['engine_bytes'] for s in self.samples if s.get('engine_bytes') is not None]
        gib = lambda v: round(v / 2**30, 3) if v is not None else None
        return dict(footprint_start_gib=gib(values[0]) if values else None,
                    footprint_peak_sampled_gib=gib(max(values)) if values else None,
                    footprint_lifetime_peak_gib=gib(max(lifetime)) if lifetime else None,
                    engine_reported_peak_gib=gib(max(engine)) if engine else None,
                    samples=len(self.samples))

# ---------------------------------------------------------------- client

def post_stream(port, body, timeout):
    request = Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ.get('BENCH_API_KEY', 'local')})
    started = time.time()
    first = None
    content, reasoning, calls, usage, extra, finish = [], [], {}, None, {}, None
    with urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'): continue
            data = line[5:].strip()
            if data == '[DONE]': break
            chunk = json.loads(data)
            usage = chunk.get('usage') or usage
            for key in ('tensorfold', 'timings'):
                if chunk.get(key): extra[key] = chunk[key]
            for choice in chunk.get('choices', []):
                delta = choice.get('delta') or {}
                text = delta.get('content')
                think = delta.get('reasoning_content') or delta.get('reasoning')
                if (text or think or delta.get('tool_calls')) and first is None:
                    first = time.time()
                if text: content.append(text)
                if think: reasoning.append(think)
                for call in delta.get('tool_calls') or []:
                    slot = calls.setdefault(call.get('index', 0), dict(id=None, name='', arguments=''))
                    slot['id'] = call.get('id') or slot['id']
                    fn = call.get('function') or {}
                    slot['name'] += fn.get('name') or ''
                    slot['arguments'] += fn.get('arguments') or ''
                finish = choice.get('finish_reason') or finish
    ended = time.time()
    return dict(content=''.join(content), reasoning=''.join(reasoning),
                tool_calls=[calls[k] for k in sorted(calls)], usage=usage or {}, extra=extra, finish=finish,
                ttft=None if first is None else first - started, seconds=ended - started)


def agent_run(engine, args, expected, seed):
    cfg = ENGINES[engine]
    messages = [{'role': 'system', 'content': 'You are a careful data analyst working in a read-only workspace. '
                 'Use the tools to inspect files; do not guess values. Work efficiently.'},
                {'role': 'user', 'content': QUESTION}]
    steps, seen, answer, started = [], set(), None, time.time()
    totals = dict(prompt=0, cached=0, completion=0, reasoning=0, tool_calls=0, invalid_tool_calls=0,
                  repeated_tool_calls=0)
    error = None
    while len(steps) < args.max_steps and answer is None:
        if time.time() - started > args.time_limit:
            error = 'time limit'; break
        body = dict(model=cfg['model'], messages=messages, tools=TOOLS, stream=True,
                    stream_options={'include_usage': True}, max_tokens=args.max_tokens,
                    temperature=args.temperature, top_p=args.top_p, seed=seed)
        try:
            reply = post_stream(cfg['port'], body, timeout=args.time_limit)
        except HTTPError as exc:
            error = f'HTTP {exc.code}: {exc.read()[:300].decode("utf-8", "replace")}'; break
        except OSError as exc:
            error = f'{type(exc).__name__}: {exc}'; break
        u = reply['usage']
        cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
        reasoning_tokens = (u.get('completion_tokens_details') or {}).get('reasoning_tokens')
        tf = reply['extra'].get('tensorfold') or {}
        step = dict(step=len(steps) + 1, seconds=round(reply['seconds'], 3),
                    ttft=None if reply['ttft'] is None else round(reply['ttft'], 3), finish=reply['finish'],
                    prompt_tokens=u.get('prompt_tokens'), cached_tokens=cached,
                    completion_tokens=u.get('completion_tokens'), reasoning_tokens=reasoning_tokens,
                    reasoning_chars=len(reply['reasoning']), content_chars=len(reply['content']),
                    tools=[c['name'] for c in reply['tool_calls']],
                    engine_runtime={k: tf.get(k) for k in ('prefill_seconds', 'time_to_first_token', 'seconds',
                        'tokens_per_second', 'acceptance_rate', 'tokens_per_round', 'drafted', 'accepted') if k in tf}
                    or reply['extra'].get('timings'))
        if step['completion_tokens'] and reply['ttft'] is not None and reply['seconds'] > reply['ttft']:
            step['decode_tok_s_client'] = round((step['completion_tokens'] - 1) / (reply['seconds'] - reply['ttft']), 2)
        steps.append(step)
        totals['prompt'] += u.get('prompt_tokens') or 0
        totals['cached'] += cached or 0
        totals['completion'] += u.get('completion_tokens') or 0
        totals['reasoning'] += reasoning_tokens or 0
        assistant = {'role': 'assistant', 'content': reply['content'] or None}
        if reply['tool_calls']:
            assistant['tool_calls'] = [{'id': c['id'] or f'call_{len(steps)}_{i}', 'type': 'function',
                'function': {'name': c['name'], 'arguments': c['arguments'] or '{}'}}
                for i, c in enumerate(reply['tool_calls'])]
        messages.append(assistant)
        if not reply['tool_calls']:
            messages.append({'role': 'user', 'content': 'Continue with the tools and finish by calling submit_answer.'})
            continue
        for call in assistant['tool_calls']:
            totals['tool_calls'] += 1
            name, raw = call['function']['name'], call['function']['arguments']
            key = (name, raw)
            if key in seen: totals['repeated_tool_calls'] += 1
            seen.add(key)
            try:
                parsed = json.loads(raw)
                if name == 'submit_answer':
                    answer = parsed.get('answer', parsed)
                    result = 'Answer recorded.'
                else:
                    result = run_tool(name, parsed)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                totals['invalid_tool_calls'] += 1
                result = f'Error: {exc}'
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': result})
    wall = time.time() - started
    verdict = grade(answer, expected) if answer is not None else dict(correct=False, matched=0,
                                                                         expected=len(expected), reason=error or 'no answer')
    return dict(wall_seconds=round(wall, 2), time_to_correct_s=round(wall, 2) if verdict['correct'] else None,
                steps=len(steps), totals=totals, verdict=verdict, answer=answer, error=error, step_log=steps,
                prompt_tokens_last=steps[-1]['prompt_tokens'] if steps else None)

# ---------------------------------------------------------------- dashboard control

def dashboard(path, body=None, timeout=10):
    request = Request(DASHBOARD + path, data=None if body is None else json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json', 'X-Sushi-Control': '1'})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_idle(engine, online, limit):
    for _ in range(limit):
        status = dashboard(f'/api/status?engine={engine}')
        if status.get('error'): raise RuntimeError(f'{engine}: {status["error"]}')
        if not status.get('busy') and bool(status.get('online')) == online: return status
        time.sleep(1)
    raise RuntimeError(f'{engine} did not become {"ready" if online else "stopped"} in {limit}s')


def fresh_engine(engine):
    """Stop every engine, then start the selected one so its lifetime peak covers only this benchmark."""
    for name in ENGINES:
        if dashboard(f'/api/status?engine={name}').get('online'):
            print(f'Stopping {name}…', flush=True)
            dashboard('/api/action', dict(action='stop', engine=name))
            wait_idle(name, False, 90)
    print(f'Starting {engine}…', flush=True)
    t = time.time()
    dashboard('/api/action', dict(action='start', engine=engine))
    wait_idle(engine, True, 1200)
    return round(time.time() - t, 1)

# ---------------------------------------------------------------- reporting

def host_info():
    info = dict(machine=platform.machine(), system=platform.system(), release=platform.release())
    if platform.system() == 'Darwin':
        for key in ('machdep.cpu.brand_string', 'hw.memsize'):
            try: info[key] = subprocess.check_output(['sysctl', '-n', key], text=True).strip()
            except (OSError, subprocess.CalledProcessError): pass
    return info


def write_report():
    rows = []
    for path in sorted(RESULTS.glob('*.json')):
        data = json.loads(path.read_text())
        for i, run in enumerate(data['runs'], 1):
            rows.append((data, run, i, path.name))
    lines = ['# Agent benchmark results', '',
             'Generated by `agent_bench.py` from the JSON files in `results/`; do not edit by hand.',
             f'Task: nightly-history ({rows[0][0]["task_sha"][:12] if rows else "—"}); memory target {MEMORY_TARGET_GIB:g} GiB.',
             '', '| Started | Engine | Run | Correct | Matched | Wall s | Steps | Tool calls (bad/rep) | '
             'Prompt / cached / out tok | Mean TTFT s | Decode tok/s | Draft acc. | Peak footprint GiB | Swap Δ MiB | File |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for data, run, i, name in rows:
        t = run['totals']
        ttfts = [s['ttft'] for s in run['step_log'] if s.get('ttft') is not None]
        rates = [s['decode_tok_s_client'] for s in run['step_log'] if s.get('decode_tok_s_client')]
        acc = [s['engine_runtime'].get('acceptance_rate') for s in run['step_log']
               if isinstance(s.get('engine_runtime'), dict) and s['engine_runtime'].get('acceptance_rate') is not None]
        mem = run['memory']
        peak = mem.get('footprint_lifetime_peak_gib') or mem.get('footprint_peak_sampled_gib')
        flag = ' ⚠' if peak and peak > MEMORY_TARGET_GIB else ''
        swap = run.get('swap_delta_mib')
        lines.append('| {} | {} | {} | {} | {}/{} | {} | {} | {} ({}/{}) | {} / {} / {} | {} | {} | {} | {} | {} | {} |'.format(
            data['started'][:16], data['engine'], i, 'yes' if run['verdict']['correct'] else 'no',
            run['verdict']['matched'], run['verdict']['expected'], run['wall_seconds'], run['steps'],
            t['tool_calls'], t['invalid_tool_calls'], t['repeated_tool_calls'], t['prompt'], t['cached'], t['completion'],
            round(sum(ttfts) / len(ttfts), 2) if ttfts else '—', round(sum(rates) / len(rates), 1) if rates else '—',
            f'{100 * sum(acc) / len(acc):.0f}%' if acc else '—',
            f'{peak}{flag}' if peak else '—', '—' if swap is None else round(swap), name))
    lines += ['', 'Peak footprint is the kernel physical footprint (Metal included) of the server process; the',
              'lifetime peak is used when available. ⚠ marks runs above the memory target. Decode tok/s is',
              'client-side (completion tokens after the first / streaming time) and includes reasoning tokens.', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--engine', choices=list(ENGINES))
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--manage', action='store_true', help='Restart the engine through the dashboard first.')
    parser.add_argument('--stop-after', action='store_true', help='Stop the engine through the dashboard at the end.')
    parser.add_argument('--max-steps', type=int, default=30)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--time-limit', type=float, default=1800)
    # TensorFold's Qwen 27B drafts greedy requests only; sampled ones decode plain. Same setting for every engine.
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--label', default='', help='Free-text note saved with the results.')
    parser.add_argument('--make-fixture', action='store_true', help='Regenerate the committed fixture and exit.')
    parser.add_argument('--report', action='store_true', help='Regenerate RESULTS.md and exit.')
    args = parser.parse_args()
    if args.make_fixture:
        make_fixture(); print(json.dumps(expected_answer(), indent=2)); return
    if args.report:
        write_report(); return
    if not args.engine:
        parser.error('--engine is required')
    expected = expected_answer()
    task_sha = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(FIXTURE.rglob('*.csv'))) +
                              QUESTION.encode()).hexdigest()
    startup = fresh_engine(args.engine) if args.manage else None
    pid = server_pid(args.engine)
    if pid is None:
        print('Warning: server PID not found; process footprint will be missing.', file=sys.stderr)
    RESULTS.mkdir(exist_ok=True)
    started = datetime.datetime.now().astimezone()
    out = RESULTS / f'{started:%Y%m%d-%H%M%S}-{args.engine}.json'
    record = dict(engine=args.engine, model=ENGINES[args.engine]['model'], started=started.isoformat(),
                  label=args.label, host=host_info(), task='nightly-history', task_sha=task_sha,
                  settings={k: getattr(args, k) for k in ('max_steps', 'max_tokens', 'time_limit', 'temperature', 'top_p')},
                  memory_target_gib=MEMORY_TARGET_GIB, startup_seconds=startup, pid=pid,
                  idle_footprint_gib=None, runs=[])
    idle = footprint(pid)[0]
    record['idle_footprint_gib'] = round(idle / 2**30, 3) if idle else None
    for i in range(args.runs):
        sampler = Sampler(args.engine, pid)
        swap_before, level_before = swap_used_mib(), memory_level()
        sampler.start()
        result = agent_run(args.engine, args, expected, seed=1000 + i)
        sampler.stop.set(); sampler.join()
        swap_after = swap_used_mib()
        result.update(memory=sampler.summary(), swap_before_mib=swap_before, swap_after_mib=swap_after,
                      swap_delta_mib=None if None in (swap_before, swap_after) else swap_after - swap_before,
                      memory_level_before=level_before, memory_level_after=memory_level(),
                      memory_trace=[{k: v for k, v in s.items() if v is not None} for s in sampler.samples[::4]])
        peak = result['memory']['footprint_lifetime_peak_gib'] or result['memory']['footprint_peak_sampled_gib']
        result['over_memory_target'] = None if peak is None else peak > MEMORY_TARGET_GIB
        record['runs'].append(result)
        out.write_text(json.dumps(record, indent=1, ensure_ascii=False))  # Saved after every run.
        print(f'[{args.engine} run {i + 1}] correct={result["verdict"]["correct"]} '
              f'matched={result["verdict"]["matched"]}/{result["verdict"]["expected"]} wall={result["wall_seconds"]}s '
              f'steps={result["steps"]} peak={peak} GiB swapΔ={result["swap_delta_mib"]} MiB'
              + (f' error={result["error"]}' if result['error'] else ''), flush=True)
    if args.stop_after and args.manage:
        dashboard('/api/action', dict(action='stop', engine=args.engine))
    write_report()
    print(f'Saved {out.relative_to(HERE.parent)}; summary in {(HERE / "RESULTS.md").relative_to(HERE.parent)}')


if __name__ == '__main__':
    main()
