#!/usr/bin/env python3
"""Overlap test on the FN 5090 lane (:8001, 2 slots): one request decoding while
another prefills a big prompt. Tails /tmp/rgbshow.status to catch the window
where green (decode) and orange (prefill) light together. Never prints keys."""
import json, secrets, threading, time, urllib.request, urllib.error

KEY = json.load(open("~/.config/strata/launch-presets/main.json"))["api_key"]
URL = "http://127.0.0.1:8001/v1/chat/completions"


def post(content, max_tokens):
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s",
                       "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0.7}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    try:
        urllib.request.urlopen(req, timeout=600)
        print(f"req done {max_tokens}tk in {time.time()-t0:.1f}s", flush=True)
    except urllib.error.HTTPError as e:
        print("ERROR", e.code, e.read()[:200], flush=True)


SALT = secrets.token_hex(8)  # unique per run: defeats prefix cache, forces real prefill
big = "One-line note.\n" + "\n".join(
    f"Note {i}: alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo {i%97} {SALT}"
    for i in range(2600))  # ~26 tokens/line ~ 65k tokens

# slot 1: long decode (~10 s), slot 2: big prefill starting 2 s later
threading.Thread(target=post, args=("Count from 1 to 2000, one per line.", 1700), daemon=True).start()
time.sleep(2.0)
threading.Thread(target=post, args=(big + "\nSummarize in 20 words.", 60), daemon=True).start()

for _ in range(30):
    time.sleep(0.5)
    with open("/tmp/rgbshow.status") as f:
        ln = f.read().strip()
    seg = ln.split("tps5090", 1)[-1]
    print("tps5090" + seg[: seg.index(" hot")], flush=True)
