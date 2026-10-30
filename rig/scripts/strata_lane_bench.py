#!/usr/bin/env python3
"""Quick decode/prefill probe for any Strata server (main :8001 or a second lane).

Sequential (or concurrent with --par), temperature 0, thinking off. Numbers come
from the response `timings` block the Strata server returns.

usage: strata_lane_bench.py PORT PRESET TAG [--n 3] [--par]
  PRESET = launch preset name whose api_key is used (main, lane2, ...)
"""
import argparse
import json
import threading
import time
import urllib.request

PROMPTS = [
    "Explain how speculative decoding works in large language models. Be detailed.",
    "Объясни простыми словами, как работает спекулятивное декодирование в языковых моделях.",
    "Write a Python function that parses an nginx access log line into a dict, with tests.",
    "Сравни PostgreSQL и ClickHouse для аналитики: плюсы, минусы, когда что выбирать.",
]


def one(port, key, prompt, out, idx):
    body = {
        "model": "qwen3.8-flash-next-iq3_s",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    wall = time.time() - t0
    tm = d.get("timings") or {}
    u = d.get("usage") or {}
    out[idx] = {
        "wall_s": round(wall, 2),
        "completion_tokens": u.get("completion_tokens"),
        "prompt_tokens": u.get("prompt_tokens"),
        "decode_tok_s": tm.get("predicted_per_second") or tm.get("decode_tok_s"),
        "prefill_tok_s": tm.get("prompt_per_second") or tm.get("prefill_tok_s"),
        "timings": tm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("preset")
    ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--par", action="store_true", help="send all prompts at once")
    a = ap.parse_args()
    key = json.load(open(f"~/.config/strata/launch-presets/{a.preset}.json"))["api_key"]
    prompts = PROMPTS[: a.n]
    res = [None] * len(prompts)
    if a.par:
        th = [threading.Thread(target=one, args=(a.port, key, p, res, i)) for i, p in enumerate(prompts)]
        [t.start() for t in th]
        [t.join() for t in th]
    else:
        for i, p in enumerate(prompts):
            one(a.port, key, p, res, i)
    for i, r in enumerate(res):
        print(a.tag, i, json.dumps({k: v for k, v in r.items() if k != "timings"}))
    print(a.tag, "timings_keys", sorted((res[0] or {}).get("timings", {}).keys()))


if __name__ == "__main__":
    main()
