#!/usr/bin/env python3
"""One A/B arm's measurements against the running Maya (:18080). Every number is the engine's own
"prompt N = R reused + X read in T ms" line from the arm's engine log.
  cold   fresh prompts (a random nonce first: nothing reusable) of --cold thousand tokens
  agent  a fresh 24K start, then appends of 0.6K / 1.2K / 3K new tokens on it (agent turns; this is
         where MAYA_DS4_PREFILL_CPU acts - chunks of at most 4K tokens)
  decode 3 answers of 300 tokens, greedy, thinking off
usage: probe.py ARM [--cold 24,128,400] [--skip-decode]
"""
import argparse
import json
import random
import re
import time
import urllib.request

U = "http://127.0.0.1:18080/v1/chat/completions"
SRC = open("/opt/rig/monitor/monitor.py").read() + open("~/maya/docs/SETTINGS.md").read()
RX = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                r"(\d+) generated in (\d+) ms")
ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("--cold", default="24,128,400")
ap.add_argument("--skip-decode", action="store_true")
a = ap.parse_args()
LOG = f"~/ds-pf/{a.arm}.engine.log"


def text(k, off=0):
    s = SRC[off % len(SRC):] + SRC
    return (s * (k * 4000 // len(s) + 1))[: k * 4000]


def lines():
    try:
        return [l for l in open(LOG, errors="replace") if RX.search(l)]
    except OSError:
        return []


def call(msgs, mt):
    n0 = len(lines())
    b = {"model": "x", "messages": msgs, "max_tokens": mt, "temperature": 0, "reasoning_effort": "none"}
    t = time.time()
    req = urllib.request.Request(U, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=7200))
    wall = time.time() - t
    ls = lines()
    for _ in range(20):
        if len(ls) > n0:
            break
        time.sleep(0.5)
        ls = lines()
    m = RX.search(ls[-1])
    return d, {"prompt": int(m.group(1)), "reused": int(m.group(2)), "read": int(m.group(3)),
               "read_s": round(int(m.group(4)) / 1000, 2), "prefill": float(m.group(5)), "gen": int(m.group(6)),
               "dec": round(int(m.group(6)) / max(1e-3, int(m.group(7)) / 1000), 1), "wall": round(wall, 1)}


def out(kind, **kw):
    print(json.dumps({"arm": a.arm, "kind": kind, **kw}), flush=True)


call([{"role": "user", "content": "Say OK."}], 4)   # warm-up: GEMM start, first expert moves
for k in [int(x) for x in a.cold.split(",") if x]:
    _, r = call([{"role": "user", "content": f"nonce {random.random()}\n{text(k)}\nSummarize in one word."}], 8)
    out("cold", k=k, **r)

msgs = [{"role": "system", "content": f"nonce {random.random()}\nYou are a code reviewer. Material:\n{text(24)}"},
        {"role": "user", "content": "Read the material. Reply OK."}]
d, r = call(msgs, 8)
out("agent-start", **r)
for i, k in enumerate((0.6, 1.2, 3.0)):
    msgs += [{"role": "assistant", "content": d["choices"][0]["message"].get("content") or "OK"},
             {"role": "user", "content": text(4, off=7919 * (i + 1))[: int(k * 4000)] + "\nAnything wrong? One line."}]
    d, r = call(msgs, 16)
    out("agent-turn", add_k=k, **r)

if not a.skip_decode:
    for q in ("Explain in detail how a modern CPU branch predictor works.",
              "Write a Python function that parses an nginx access log line into a dict, with tests.",
              "Объясни подробно, как устроена система охлаждения компьютера.",
              "Explain how TCP congestion control works, step by step.",
              "Write a bash script that rotates logs older than 7 days and compresses them, with comments.",
              "Расскажи историю Белграда в пяти абзацах."):
        _, r = call([{"role": "user", "content": q}], 400)
        out("decode", q=q[:20], **r)
