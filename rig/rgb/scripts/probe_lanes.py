#!/usr/bin/env python3
"""Fire one small chat request on each FN 3090 lane (:8003, :8004) and tail
/tmp/rgbshow.status while they decode. Reads Strata keys from the lane presets;
never prints them. Usage: probe_lanes.py [max_tokens]"""
import json, sys, threading, time, urllib.request

MT = int(sys.argv[1]) if len(sys.argv) > 1 else 400
LANES = [("l2", 8003, "lane2.json"), ("l3", 8004, "lane3.json")]


def key_for(preset):
    p = json.load(open(f"~/.config/strata/launch-presets/{preset}"))
    return p["api_key"]


def fire(name, port, preset):
    body = json.dumps({
        "model": "qwen3.8-flash-next-iq3_s",
        "messages": [{"role": "user", "content": "Count slowly from 1 to 300, one number per line."}],
        "max_tokens": MT, "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key_for(preset)}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        u = d.get("usage", {})
        print(f"{name} done: completion_tokens={u.get('completion_tokens')}", flush=True)
    except Exception as e:
        print(f"{name} ERROR: {e}", flush=True)


for args in LANES:
    threading.Thread(target=fire, args=args, daemon=True).start()

for _ in range(30):
    time.sleep(0.7)
    try:
        print(open("/tmp/rgbshow.status").read().strip(), flush=True)
    except Exception:
        pass
