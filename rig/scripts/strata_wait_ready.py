#!/usr/bin/env python3
"""Wait until a Strata server answers a real chat (health goes green too early).

usage: strata_wait_ready.py PORT PRESET [TIMEOUT_S]
Reads the API key from the launch preset file itself (never pass keys on a command line).
"""
import json
import sys
import time
import urllib.request

port, preset = int(sys.argv[1]), sys.argv[2]
deadline = time.time() + (int(sys.argv[3]) if len(sys.argv) > 3 else 1200)
key = json.load(open(f"~/.config/strata/launch-presets/{preset}.json"))["api_key"]
body = json.dumps({
    "model": "qwen3.8-flash-next-iq3_s",
    "messages": [{"role": "user", "content": "Say OK"}],
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
t0 = time.time()
last = ""
while time.time() < deadline:
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
        msg = d["choices"][0]["message"]
        if msg.get("content") or msg.get("reasoning_content"):
            print(f"READY after {time.time() - t0:.0f}s: {str(msg.get('content'))[:60]!r}")
            sys.exit(0)
    except Exception as e:  # noqa: BLE001 - keep polling on any error
        last = repr(e)[:160]
    time.sleep(10)
print("TIMEOUT", last)
sys.exit(1)
