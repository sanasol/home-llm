#!/usr/bin/env python3
"""Pilot check for the qwen3.8-27b pool on a LiteLLM instance (default :8011).
Reports which deployment (x-litellm-model-id) served each request.
usage: pool_routing_check.py [port] [mode]   mode: spread | dead
"""
import json, sys, threading, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8011"
MODE = sys.argv[2] if len(sys.argv) > 2 else "spread"
K = open("/opt/litellm/master.key").read().strip()


def call(first, sid, turn2=False):
    msgs = [{"role": "user", "content": first}]
    if turn2:
        msgs += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "and one more word"}]
    body = {"model": "qwen3.8-27b", "messages": msgs, "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bear" + "er " + K,
                                  "x-litellm-session-id": sid})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.headers.get("x-litellm-model-id"), r.status
    except Exception as e:  # noqa: BLE001
        return None, repr(e)[:120]


res = {}
def run(i, turn2):
    res[(i, turn2)] = call(f"conversation {i}: name a colour", "parent-root-123", turn2)

for turn2 in (False, True):
    ths = [threading.Thread(target=run, args=(i, turn2)) for i in range(6)]
    [t.start() for t in ths]
    [t.join() for t in ths]
for i in range(6):
    print(i, "turn1", res[(i, False)], "turn2", res[(i, True)])
