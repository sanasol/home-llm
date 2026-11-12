#!/usr/bin/env python3
"""PR #959 (MAYA_DS4_PREFILL_CPU) measurement against the running Maya on :18080, same protocol as the PR:
  fresh   prompts of ~330 / ~600 / ~1,600 tokens, nothing reusable (a nonce first), first token only
  agent   a ~20.6K-token start read once, then 6 appends of 500-3,200 new tokens on it (an agent's turns)
  logits  the first answer token's top-10 log-probabilities of 3 fixed prompts (fixed text, no nonce) -> KL off/on
  decode  3 answers of 300 tokens, greedy, thinking off
Timings are the engine's own "prompt N = R reused + X read in T ms" lines (arm log). One JSON line per measurement.
usage: pr959_probe.py ARM ROUND
"""
import json
import random
import re
import sys
import time
import urllib.request

U = "http://127.0.0.1:18080/v1/chat/completions"
ARM, ROUND = sys.argv[1], int(sys.argv[2])
LOG = f"~/ds-pf/{ARM}.engine.log"
SRC = open("/opt/rig/monitor/monitor.py").read() + open("~/maya/docs/SETTINGS.md").read()
RX = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                r"(\d+) generated in (\d+) ms")


def text(chars, off):
    s = SRC[off % len(SRC):] + SRC
    return (s * (chars // len(s) + 1))[:chars]


def lines():
    try:
        return [l for l in open(LOG, errors="replace") if RX.search(l)]
    except OSError:
        return []


def call(msgs, mt, logprobs=False):
    n0 = len(lines())
    b = {"model": "x", "messages": msgs, "max_tokens": mt, "temperature": 0, "reasoning_effort": "none"}
    if logprobs:
        b.update(logprobs=True, top_logprobs=10)
    req = urllib.request.Request(U, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    wall = time.time() - t
    ls = lines()
    for _ in range(20):
        if len(ls) > n0:
            break
        time.sleep(0.5)
        ls = lines()
    m = RX.search(ls[-1])
    return d, {"prompt": int(m.group(1)), "reused": int(m.group(2)), "read": int(m.group(3)),
               "read_s": round(int(m.group(4)) / 1000, 3), "gen": int(m.group(6)),
               "dec": round(int(m.group(6)) / max(1e-3, int(m.group(7)) / 1000), 2), "wall": round(wall, 2)}


def out(kind, **kw):
    print(json.dumps({"round": ROUND, "arm": ARM, "kind": kind, **kw}, ensure_ascii=False), flush=True)


rng = random.Random()
call([{"role": "user", "content": "Say OK."}], 4)                                   # warm-up
call([{"role": "user", "content": f"nonce {rng.random()}\n{text(2000, 11)}\nOK?"}], 4)

for chars in (1300, 2400, 6400):                                                     # ~330 / 600 / 1,600 tokens
    _, r = call([{"role": "user", "content": f"nonce {rng.random()}\n{text(chars, rng.randrange(99999))}\nOne word."}], 1)
    out("fresh", chars=chars, **r)

msgs = [{"role": "system", "content": f"nonce {rng.random()}\nYou review this code base. Material:\n{text(82000, 5)}"},
        {"role": "user", "content": "Read it. Reply OK."}]
d, r = call(msgs, 4)
out("agent-start", **r)
for i, chars in enumerate((2000, 2400, 2800, 4600, 12500, 12900)):                  # ~500 .. 3,200 new tokens
    msgs += [{"role": "assistant", "content": d["choices"][0]["message"].get("content") or "OK"},
             {"role": "user", "content": text(chars, 7919 * (i + 1)) + "\nAnything wrong here? One line."}]
    d, r = call(msgs, 8)
    out("agent-turn", chars=chars, **r)

for i, chars in enumerate((1300, 2400, 6400)):                                       # fixed prompts: comparable logits
    d, r = call([{"role": "user", "content": text(chars, 31337 * (i + 1)) + "\nName the language of this text."}], 1,
                logprobs=True)
    lp = ((d["choices"][0].get("logprobs") or {}).get("content") or [{}])[0]
    top = {t["token"]: t["logprob"] for t in (lp.get("top_logprobs") or [])}
    out("logits", chars=chars, first=lp.get("token"), top=top, **r)

for q in ("Explain in detail how a modern CPU branch predictor works.",
          "Write a Python function that parses an nginx access log line into a dict, with tests.",
          "Explain how TCP congestion control works, step by step."):
    _, r = call([{"role": "user", "content": q}], 300)
    out("decode", q=q[:24], **r)

stats = [l.strip() for l in open(LOG, errors="replace") if "DS4_PREFILL_CPU gpu=" in l]
out("stats", lines=stats[-12:])
