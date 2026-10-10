#!/usr/bin/env python3
"""Long cached chat benchmark: per-turn prefill of the new tokens, TTFT, decode tok/s and process memory.

Builds a synthetic conversation of about --history-tokens tokens (CSV runs and log text from the committed
fixture plus canned assistant turns), sends it once (warm-up; cold prefill is recorded but not the point),
then continues the same chat for --turns turns. Each turn appends the real previous reply and a short new
user message, so a working prefix cache only has to prefill a few hundred new tokens.

Standard library only. Results go to bench/results/longchat-*.json and bench/LONGCHAT.md.

    python3 mac-local/bench/longchat_bench.py --port 12346 --model Bonsai-2-27B --label bonsai-llamacpp
"""
import argparse
import datetime
import json
import random
import subprocess
import time
from pathlib import Path

import agent_bench as ab

HERE = Path(__file__).resolve().parent
FOLLOW_UPS = [
    'Go back to the 2026-10-03 run file. Walk through every job that did not end with status ok, one paragraph each: '
    'what the message says happened, the exit code, and whether it looks transient. Then summarise the pattern.',
    'Now compare that with 2026-10-06. Which hosts appear in failures on both nights? Explain your reasoning step by '
    'step and list the rows you used.',
    'Write a Python function that parses one of these CSV files regardless of column order and returns failures per '
    'job. Include docstrings and explain the edge cases (retries, timeout, skipped).',
    'Draft a short incident note for the platform team about the three noisiest jobs this week: impact, likely '
    'cause from the messages, and next steps. Use headings.',
    'Review your previous answers in this conversation for mistakes about column order or retries and correct '
    'anything wrong. Quote the rows that prove each correction.',
    'Propose monitoring thresholds for each team based on these runs, with a table and a justification per row.',
]


def build_history(target_tokens, seed=7):
    """Messages totalling roughly target_tokens (estimated at 3.2 chars a token for CSV-heavy text)."""
    rng = random.Random(seed)
    files = sorted((ab.FIXTURE / 'runs').glob('*.csv'))
    jobs = (ab.FIXTURE / 'jobs.csv').read_text()
    messages = [{'role': 'system', 'content': 'You are a careful operations analyst. Answer from the data in this '
                 'conversation, cite rows, and say when something is not in the data.'},
                {'role': 'user', 'content': 'Here is the job ownership table:\n\n' + jobs}]
    messages.append({'role': 'assistant', 'content': 'Noted: 30 jobs across six teams. Send the nightly run files.'})
    budget = target_tokens * 3.2
    used = sum(len(m['content']) for m in messages)
    turn = 0
    while used < budget:
        file = files[turn % len(files)]
        text = file.read_text()
        if turn >= len(files):  # Later passes: reshuffled rows so the prefix is not trivially repetitive.
            head, *rows = text.splitlines()
            rng.shuffle(rows)
            text = '\n'.join([head] + rows)
        user = f'Run file {file.name} (pass {turn // len(files) + 1}):\n\n{text}'
        reply = (f'Read {file.name}: {len(text.splitlines()) - 1} rows. The header order is '
                 f'{text.splitlines()[0]}. I will keep the failed and timeout rows in mind for later questions.')
        messages += [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': reply}]
        used += len(user) + len(reply)
        turn += 1
    return messages


def turn_request(args, messages, seed):
    body = dict(model=args.model, messages=messages, stream=True, stream_options={'include_usage': True},
                max_tokens=args.max_tokens, temperature=args.temperature, top_p=0.95, top_k=20, seed=seed,
                cache_prompt=True, chat_template_kwargs={'reasoning_effort': 'medium'})
    if args.ignore_eos:
        body['ignore_eos'] = True
    return ab.post_stream(args.port, body, timeout=args.timeout)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--history-tokens', type=int, default=30000)
    p.add_argument('--turns', type=int, default=5)
    p.add_argument('--max-tokens', type=int, default=600)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--ignore-eos', action='store_true')
    p.add_argument('--timeout', type=float, default=3600)
    p.add_argument('--note', default='')
    args = p.parse_args()
    out = subprocess.check_output(['lsof', '-ti', f'tcp:{args.port}', '-sTCP:LISTEN'], text=True)
    pid = int(out.split()[0])
    messages = build_history(args.history_tokens)
    sampler = ab.Sampler('none', pid)
    sampler.start()
    swap_before = ab.swap_used_mib()
    started = datetime.datetime.now().astimezone()
    record = dict(label=args.label, model=args.model, port=args.port, pid=pid, started=started.isoformat(),
                  note=args.note, host=ab.host_info(), settings=vars(args), turns=[])
    path = HERE / 'results' / f'longchat-{started:%Y%m%d-%H%M%S}-{args.label}.json'
    path.parent.mkdir(exist_ok=True)
    for i in range(args.turns + 1):
        if i:
            messages.append({'role': 'user', 'content': FOLLOW_UPS[(i - 1) % len(FOLLOW_UPS)]})
        else:
            messages.append({'role': 'user', 'content': 'Summarise which jobs failed most often across all files.'})
        t0 = time.time()
        reply = turn_request(args, messages, seed=100 + i)
        u = reply['usage']
        prompt = u.get('prompt_tokens')
        cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
        timings = reply['extra'].get('timings') or {}
        tf = reply['extra'].get('tensorfold') or {}
        if cached is None and timings.get('cache_n') is not None:
            cached = timings['cache_n']
        new = None if prompt is None or cached is None else prompt - cached
        decode_s = reply['seconds'] - (reply['ttft'] or 0)
        comp = u.get('completion_tokens') or 0
        current, peak = ab.footprint(pid)
        turn = dict(turn=i, prompt_tokens=prompt, cached_tokens=cached, new_prompt_tokens=new,
                    ttft_s=round(reply['ttft'], 3) if reply['ttft'] else None,
                    prefill_tok_s=round(new / reply['ttft'], 1) if new and reply['ttft'] else None,
                    completion_tokens=comp, decode_s=round(decode_s, 3),
                    decode_tok_s=round((comp - 1) / decode_s, 2) if comp > 1 and decode_s > 0 else None,
                    wall_s=round(time.time() - t0, 2), finish=reply['finish'],
                    reasoning_chars=len(reply['reasoning']), content_chars=len(reply['content']),
                    server_timings=timings or None,
                    server_tensorfold={k: tf.get(k) for k in ('prefill_seconds', 'time_to_first_token', 'seconds',
                        'tokens_per_second', 'acceptance_rate', 'tokens_per_round', 'drafts', 'sampling') if k in tf} or None,
                    footprint_gib=round(current / 2**30, 2) if current else None,
                    lifetime_peak_gib=round(peak / 2**30, 2) if peak else None)
        record['turns'].append(turn)
        print(json.dumps({k: turn[k] for k in ('turn', 'prompt_tokens', 'new_prompt_tokens', 'ttft_s', 'prefill_tok_s',
                                               'completion_tokens', 'decode_tok_s', 'footprint_gib', 'lifetime_peak_gib')}),
              flush=True)
        # Keep the real reply (content only, as clients do) so the next turn extends the cached prefix.
        messages.append({'role': 'assistant', 'content': reply['content'] or '(no answer)'})
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    sampler.stop.set()
    sampler.join()
    record.update(memory=sampler.summary(), swap_before_mib=swap_before, swap_after_mib=ab.swap_used_mib())
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    report()
    print('saved', path.name)


def report():
    rows = ['# Long cached chat results', '',
            'Generated by `longchat_bench.py`. Turn 0 is the cold load of the history; turns 1+ continue the same chat.',
            '"New tok" is prompt minus cached tokens as the server reports them. Decode tok/s is client-side and',
            'includes reasoning tokens. Peak is the server process lifetime physical footprint (Metal included).', '',
            '| Started | Label | History tok | Cold TTFT s | Turns | New tok (med) | TTFT s (med) | Decode tok/s (med / min) | '
            'Accept. | Peak GiB | Swap Δ MiB |', '|---|---|---|---|---|---|---|---|---|---|---|']
    def med(v):
        v = sorted(x for x in v if x is not None)
        return v[len(v) // 2] if v else None
    for path in sorted((HERE / 'results').glob('longchat-*.json')):
        d = json.loads(path.read_text())
        t = d['turns']
        if not t: continue
        warm = t[1:]
        dec = [x['decode_tok_s'] for x in warm if x['decode_tok_s']]
        acc = [x['server_tensorfold'].get('acceptance_rate') for x in warm
               if x.get('server_tensorfold') and x['server_tensorfold'].get('acceptance_rate') is not None]
        mem = d.get('memory') or {}
        peak = mem.get('footprint_lifetime_peak_gib') or max((x['lifetime_peak_gib'] or 0) for x in t) or None
        swap = None if d.get('swap_after_mib') is None or d.get('swap_before_mib') is None else \
            round(d['swap_after_mib'] - d['swap_before_mib'])
        rows.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            d['started'][:16], d['label'], t[0]['prompt_tokens'], t[0]['ttft_s'], len(warm),
            med([x['new_prompt_tokens'] for x in warm]), med([x['ttft_s'] for x in warm]),
            f'{med(dec)} / {min(dec)}' if dec else '—', f'{100 * sum(acc) / len(acc):.0f}%' if acc else '—',
            peak, '—' if swap is None else swap))
    (HERE / 'LONGCHAT.md').write_text('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
