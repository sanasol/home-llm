#!/usr/bin/env python3
"""Dump the `live` block of Strata /metrics on a lane while it decodes."""
import json, threading, time, urllib.request

PORT = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 8003
PRESET = {8003: "lane2.json", 8004: "lane3.json"}.get(PORT, "lane2.json")
KEY = json.load(open(f"~/.config/strata/launch-presets/{PRESET}"))["api_key"]


def fire():
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s",
                       "messages": [{"role": "user", "content": "Count from 1 to 500, one per line."}],
                       "max_tokens": 900, "temperature": 0.7}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    try:
        urllib.request.urlopen(req, timeout=120)
        print("request done", flush=True)
    except Exception as e:
        print("ERROR", e, flush=True)


threading.Thread(target=fire, daemon=True).start()
for i in range(6):
    time.sleep(1.2)
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/metrics",
                                 headers={"Authorization": f"Bearer {KEY}"})
    d = json.loads(urllib.request.urlopen(req, timeout=5).read())
    live = d.get("live") or {}
    print(f"--- t={i} live keys={sorted(live.keys())}")
    print(json.dumps({k: v for k, v in live.items() if k != "requests"}, default=str)[:800], flush=True)
