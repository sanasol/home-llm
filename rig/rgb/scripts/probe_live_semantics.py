#!/usr/bin/env python3
"""Probe Strata live-block semantics on the 5090 under concurrent decode+prefill.
Dumps tok_s/tok_s_mean/prefill_tok_s_mean/prompt_read at 2 Hz. Never prints keys."""
import json, secrets, threading, time, urllib.request

KEY = json.load(open("~/.config/strata/launch-presets/main.json"))["api_key"]
H = {"Authorization": f"Bearer {KEY}"}
URL = "http://127.0.0.1:8001/v1/chat/completions"


def req(payload):
    r = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                               headers={**H, "Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=600).read()


def dec():
    try:
        req({"model": "qwen3.8-flash-next-iq3_s",
             "messages": [{"role": "user", "content": "count to 1700 in words"}],
             "max_tokens": 1700})
        print("DEC DONE", flush=True)
    except Exception as e:
        print("DEC ERR", e, flush=True)


SALT = secrets.token_hex(8)
big = "note\n" + "\n".join(f"line {i} {SALT} " + "alpha " * 12 for i in range(2600))


def pre():
    try:
        req({"model": "qwen3.8-flash-next-iq3_s",
             "messages": [{"role": "user", "content": big}, {"role": "user", "content": "hi"}],
             "max_tokens": 5})
        print("PRE DONE", flush=True)
    except Exception as e:
        print("PRE ERR", e, flush=True)


threading.Thread(target=dec).start()
time.sleep(2)
threading.Thread(target=pre).start()
for i in range(40):
    d = json.loads(urllib.request.urlopen(
        urllib.request.Request("http://127.0.0.1:8001/metrics", headers=H), timeout=5).read())
    l = d.get("live", {})
    g = l.get
    print(f"{i:2d} state={g('state')} tok_s={g('tok_s')} tok_mean={g('tok_s_mean')} "
          f"win={g('tok_s_window_s')} pre_mean={g('prefill_tok_s_mean')} read={g('prompt_read')} "
          f"run={g('running')} par={g('parallel')}", flush=True)
    time.sleep(0.5)
