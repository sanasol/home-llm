#!/usr/bin/env python3
"""Cold-prefill probe for a Strata server: unique random prefix + N K tokens of filler, max_tokens 16.

usage: strata_lane_prefill.py PORT PRESET KTOK [KTOK ...]
"""
import json
import random
import sys
import time
import urllib.request

port, preset = int(sys.argv[1]), sys.argv[2]
key = json.load(open(f"~/.config/strata/launch-presets/{preset}.json"))["api_key"]
src = open("/opt/rig/monitor/monitor.py").read()
for k in sys.argv[3:]:
    want_chars = int(k) * 1000 * 3  # ~3 chars/token on Python source
    body_txt = (src * (want_chars // len(src) + 1))[:want_chars]
    salt = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(40))
    msg = f"[{salt}] Summarise this file in one sentence.\n\n{body_txt}"
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": msg}],
                       "max_tokens": 16, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    tm = d.get("timings") or {}
    print(f"port {port} ~{k}K: prompt_n={tm.get('prompt_n')} cache_n={tm.get('cache_n')} "
          f"prefill={tm.get('prompt_per_second')} t/s wall={time.time() - t:.1f}s", flush=True)
