#!/usr/bin/env python3
"""Whole-rig capacity bench: N concurrent requests over the Strata lanes, direct ports.

usage: rig_capacity_bench.py TAG JOB [JOB ...]
  JOB = PORT:PROMPT_KTOK:MAX_TOKENS   (PROMPT_KTOK 0 = short question; >0 = unique cold filler of ~K tokens)
All jobs start at once. Per job: prompt_n, cache_n, prefill t/s, decode t/s, queue/first-token time, wall.
Aggregate: sum of generated tokens / batch wall, and generated tokens / (decode window) where decode window =
latest end - earliest decode start. Key read from the Strata preset (never on the command line).
"""
import json
import random
import sys
import threading
import time
import urllib.request

KEY = json.load(open("~/.config/strata/launch-presets/main.json"))["api_key"]
SRC = open("/opt/rig/monitor/monitor.py").read()
QUESTIONS = [
    "Explain how speculative decoding works in large language models, in detail.",
    "Write a detailed guide to choosing a NAS for a home lab, with trade-offs.",
    "Describe the history of the Internet from ARPANET to today, in depth.",
    "Explain TCP congestion control algorithms (Reno, CUBIC, BBR) with examples.",
    "Write a long technical essay on database indexing strategies.",
    "Explain how GPUs execute neural networks, from kernels to memory hierarchy.",
    "Describe how a modern CPU pipeline works, including branch prediction.",
    "Write a detailed comparison of Rust and Go for backend services.",
]


def run(i, port, ktok, max_tokens, out, t0):
    salt = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(32))
    if ktok:
        n = int(ktok * 1000 * 3.05)
        body_txt = (SRC * (n // len(SRC) + 1))[:n]
        msg = f"[{salt}] Read this file, then write a detailed review of its design.\n\n{body_txt}"
    else:
        msg = f"[{salt}] " + QUESTIONS[i % len(QUESTIONS)]
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": msg}],
                       "max_tokens": max_tokens, "temperature": 0.7, "min_tokens": max_tokens,
                       "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    s = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.load(r)
        e = time.time()
        tm = d.get("timings") or {}
        dec_ms = tm.get("predicted_ms") or 0
        out[i] = {"port": port, "ktok": ktok, "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
                  "gen": tm.get("predicted_n"), "prefill_tps": tm.get("prompt_per_second"),
                  "decode_tps": tm.get("predicted_per_second"), "start": s - t0, "end": e - t0,
                  "decode_start": e - t0 - dec_ms / 1000.0, "wall": e - s}
    except Exception as ex:  # noqa: BLE001
        out[i] = {"port": port, "ktok": ktok, "error": repr(ex)[:200]}


def main():
    tag, jobs = sys.argv[1], [j.split(":") for j in sys.argv[2:]]
    out = [None] * len(jobs)
    t0 = time.time()
    th = [threading.Thread(target=run, args=(i, int(p), float(k), int(m), out, t0)) for i, (p, k, m) in enumerate(jobs)]
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.time() - t0
    ok = [r for r in out if r and "error" not in r]
    for r in out:
        print(tag, json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}))
    gen = sum(r["gen"] or 0 for r in ok)
    if ok:
        win = max(r["end"] for r in ok) - min(r["decode_start"] for r in ok)
        print(tag, f"SUMMARY jobs={len(jobs)} ok={len(ok)} wall={wall:.1f}s gen={gen} "
                   f"agg_gen_per_wall={gen / wall:.0f} t/s agg_gen_per_decode_window={gen / win:.0f} t/s "
                   f"sum_of_stream_decode={sum(r['decode_tps'] or 0 for r in ok):.0f} t/s")


if __name__ == "__main__":
    main()
