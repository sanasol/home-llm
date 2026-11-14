#!/usr/bin/env python3
"""Small sequential refusal-evaluation client. Saves responses, never executes them or auto-scores refusal."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'harmbench': 'https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv',
    'xstest': 'https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv',
}


def load_env(path):
    """Read literal KEY=value lines; no shell expansion or command execution."""
    if not path.exists():
        return
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].lstrip()
        if '=' not in line:
            raise ValueError(f'Invalid .env line {number}')
        name, value = line.split('=', 1)
        name, value = name.strip(), value.strip()
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            raise ValueError(f'Invalid .env variable on line {number}')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        os.environ.setdefault(name, value)


def profile_config(name):
    prefix = name.upper()
    defaults = {
        'sushi': ('http://127.0.0.1:12453/v1/chat/completions', 'heretic', 'template'),
        'rig': ('http://127.0.0.1:8010/v1/chat/completions', 'your-model-name', 'template'),
        'deepseek': ('https://api.deepseek.com/chat/completions', 'deepseek-flash', 'deepseek'),
    }
    url, model, mode = defaults[name]
    return {'url': os.environ.get(prefix + '_URL', url), 'model': os.environ.get(prefix + '_MODEL', model),
            'thinking_mode': os.environ.get(prefix + '_THINKING_MODE', mode),
            'key': os.environ.get(prefix + '_API_KEY', '')}


def normalize(rows, dataset, labels):
    required = {'Behavior', 'BehaviorID', 'FunctionalCategory', 'SemanticCategory'} if dataset == 'harmbench' else {'id', 'prompt', 'type', 'label'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Unexpected {dataset} CSV columns')
    result = []
    for row in rows:
        if dataset == 'harmbench':
            if row['FunctionalCategory'].strip() != 'standard':
                continue
            item = {'id': row['BehaviorID'].strip(), 'category': row['SemanticCategory'].strip(), 'behavior': row['Behavior'], 'label': 'unsafe'}
        else:
            if labels != 'all' and row['label'].strip() != labels:
                continue
            item = {'id': row['id'].strip(), 'category': row['type'].strip(), 'behavior': row['prompt'], 'label': row['label'].strip()}
        if not all(item[k] for k in ('id', 'category', 'behavior')):
            raise ValueError('Empty dataset identity/category/prompt')
        result.append(item)
    if len({r['id'] for r in result}) != len(result):
        raise ValueError('Duplicate dataset IDs')
    return result


def select(rows, per_category, seed, limit=0):
    groups = {}
    for row in rows:
        groups.setdefault(row['category'], []).append(row)
    rng = random.Random(seed)
    chosen = []
    for category in sorted(groups):
        group = sorted(groups[category], key=lambda r: r['id'])
        rng.shuffle(group)
        chosen.extend(group if per_category == 0 else group[:per_category])
    return chosen[:limit] if limit else chosen


def payload(model, prompt, tokens, mode):
    result = {'model': model, 'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': prompt}],
              'temperature': 0, 'max_tokens': tokens, 'stream': False}
    if mode == 'template':
        result['chat_template_kwargs'] = {'thinking_mode': 'chat', 'enable_thinking': False}
    elif mode == 'deepseek':
        result['thinking'] = {'type': 'disabled'}
    elif mode != 'none':
        raise ValueError('thinking-mode must be template, deepseek or none')
    return result


def validate_url(url):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('Use an HTTP(S) endpoint without embedded credentials, query or fragment')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Endpoint redirect refused; configure the final endpoint explicitly')


def redact(value, secret):
    if not secret:return value
    if isinstance(value, str):return value.replace(secret, '[REDACTED]')
    if isinstance(value, list):return [redact(v, secret) for v in value]
    if isinstance(value, dict):return {redact(k, secret): redact(v, secret) for k, v in value.items()}
    return value


def safe_error(error, secret):
    text = str(error)
    return text.replace(secret, '[REDACTED]') if secret else text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=('sushi', 'rig', 'deepseek'), default='sushi')
    p.add_argument('--env', type=Path, default=ROOT / '.env')
    p.add_argument('--dataset', choices=tuple(SOURCES), default='harmbench')
    p.add_argument('--csv', type=Path)
    p.add_argument('--labels', choices=('safe', 'unsafe', 'all'), default='all', help='XSTest only')
    p.add_argument('--per-category', type=int, choices=(0, 1, 2), default=2, help='0=all, default at most2/category')
    p.add_argument('--limit', type=int, default=0, help='Optional total cap after category sampling')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--selection', type=Path, help='Reuse an exact saved selection across endpoints')
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--model'); p.add_argument('--url')
    p.add_argument('--thinking-mode', choices=('template', 'deepseek', 'none'))
    p.add_argument('--out', type=Path)
    p.add_argument('--execute', action='store_true', help='Without this flag, only save/print selection')
    a = p.parse_args()
    if a.max_tokens < 1 or a.limit < 0 or a.timeout < 1:
        p.error('Positive tokens/timeout and nonnegative limit required')
    load_env(a.env)
    cfg = profile_config(a.profile)
    cfg.update({k: v for k, v in {'url': a.url, 'model': a.model, 'thinking_mode': a.thinking_mode}.items() if v is not None})
    validate_url(cfg['url'])
    if a.selection:
        selection = json.loads(a.selection.read_text())
        cases = selection['cases']
        dataset, dataset_sha = selection['dataset'], selection['dataset_sha256']
        if not cases or len({r['id'] for r in cases}) != len(cases) or not all(all(r.get(k) for k in ('id','category','behavior')) for r in cases):
            raise ValueError('Invalid saved selection')
    else:
        source = a.csv or ROOT / 'data' / (a.dataset + '.csv')
        if not source.exists():
            if a.csv:
                raise FileNotFoundError(source)
            source.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(SOURCES[a.dataset], timeout=60) as response:
                raw = response.read(10 * 1024**2 + 1)
            if len(raw) > 10 * 1024**2:
                raise ValueError('Dataset exceeds10MiB limit')
            source.write_bytes(raw)
        raw = source.read_bytes()
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        cases = select(normalize(rows, a.dataset, a.labels), a.per_category, a.seed, a.limit)
        dataset, dataset_sha = a.dataset, hashlib.sha256(raw).hexdigest()
        selection = {'dataset': dataset, 'dataset_sha256': dataset_sha, 'source_url': SOURCES[dataset], 'seed': a.seed,
                     'per_category': a.per_category, 'labels': a.labels, 'cases': cases}
    if not cases:
        raise ValueError('Selection is empty')
    out = a.out or ROOT / 'results' / f'{a.profile}-{dataset}-{a.max_tokens}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    selection_path = out.with_suffix('.selection.json')
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError('Selection differs; choose a fresh --out')
    else:
        selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
    selection_sha = hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    settings = {'url': cfg['url'], 'model': cfg['model'], 'thinking_mode': cfg['thinking_mode'], 'temperature': 0,
                'max_tokens': a.max_tokens, 'dataset': dataset, 'dataset_sha256': dataset_sha, 'selection_sha256': selection_sha}
    print(json.dumps({'profile': a.profile, 'model': cfg['model'], 'count': len(cases), 'categories': dict(Counter(r['category'] for r in cases)), 'selection': str(selection_path), 'output': str(out)}, ensure_ascii=False, indent=2))
    if not a.execute:
        print('Plan only. Inspect selection; add --execute to send requests. API calls may cost money.')
        return
    if a.profile in ('rig', 'deepseek') and not cfg['key']:
        raise ValueError(f'Missing {a.profile.upper()}_API_KEY in environment or .env')
    completed = set()
    if out.exists():
        for number, line in enumerate(out.read_text().splitlines(), 1):
            if not line.strip():continue
            try: record = json.loads(line)
            except json.JSONDecodeError:raise ValueError(f'Invalid output JSON line{number}; inspect before resuming')
            if record.get('settings') != settings:raise ValueError('Output settings changed; use another --out')
            if 'response' in record:completed.add(record['id'])
    for i, case in enumerate(cases, 1):
        if case['id'] in completed:
            print(f'[{i}/{len(cases)}] {case["id"]}: skip', flush=True);continue
        body = payload(cfg['model'], case['behavior'], a.max_tokens, cfg['thinking_mode'])
        headers = {'Content-Type': 'application/json'}
        if cfg['key']:headers['Authorization'] = 'Bearer ' + cfg['key']
        record = {**case, 'profile': a.profile, 'model': cfg['model'], 'dataset': dataset, 'dataset_sha256': dataset_sha,
                  'max_tokens': a.max_tokens, 'settings': settings}
        started = time.monotonic()
        try:
            req = urllib.request.Request(cfg['url'], data=json.dumps(body).encode(), headers=headers)
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=a.timeout) as response:
                result = redact(json.load(response), cfg['key'])
            choice = result['choices'][0]
            if not isinstance(choice.get('message'), dict):raise ValueError('Response lacks assistant message')
            record['response'] = result
            record['needs_review'] = 'truncated' if choice.get('finish_reason') == 'length' else 'completed-answer-unscored'
            record['reasoning_present'] = bool(choice['message'].get('reasoning_content') or choice['message'].get('reasoning'))
            status = choice.get('finish_reason', 'unknown')
        except Exception as error:
            record['error'] = safe_error(error, cfg['key']);status = 'ERROR'
        record['seconds'] = round(time.monotonic() - started, 3)
        with out.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n');f.flush()
        print(f'[{i}/{len(cases)}] {case["category"]}/{case["id"]}: {status}, {record["seconds"]}s', flush=True)
        if 'error' in record:raise RuntimeError('Request failed; details saved without API key. Rerun retries unfinished case.')


if __name__ == '__main__':
    try: main()
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        message = str(e)
        for name in ('SUSHI_API_KEY', 'RIG_API_KEY', 'DEEPSEEK_API_KEY'):
            if os.environ.get(name):message = message.replace(os.environ[name], '[REDACTED]')
        print(message, file=sys.stderr);sys.exit(1)
