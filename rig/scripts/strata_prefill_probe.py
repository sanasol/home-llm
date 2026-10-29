#!/usr/bin/env python3
"""Cold-prefill probe for Strata :8001: fresh random-prefixed long prompts, max_tokens 1.

Reports prompt tokens, prompt_ms and prefill t/s from /metrics?requests=all.
usage: strata_prefill_probe.py TAG [SIZES_K ...]   (default sizes 8 32 64 K tokens approx)
"""
import json
import random
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8001"
KEY = json.load(open("~/.config/strata/launch-presets/main.json"))["api_key"]
HDR = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
SRC = open("/opt/rig/monitor/monitor.py", encoding="utf-8").read()


def call(path, body=None):
    req = urllib.request.Request(URL + path, headers=HDR,
                                 data=json.dumps(body).encode() if body else None)
    return json.load(urllib.request.urlopen(req, timeout=900))


def main():
    tag = sys.argv[1]
    sizes = [int(x) for x in sys.argv[2:]] or [8, 32, 64]
    for k in sizes:
        while call("/metrics").get("live", {}).get("running"):
            time.sleep(2)
        # ~4 chars/token for code; random nonce up front defeats prefix reuse
        text = (SRC * (k * 4000 // len(SRC) + 1))[: k * 4000]
        prompt = f"nonce {random.random()}\n```python\n{text}\n```\nSummarize this file in one line."
        before = {r["id"] for r in call("/metrics?requests=all")["requests"]}
        call("/v1/chat/completions", {"model": "qwen3.8-flash-next-iq3_s", "max_tokens": 1,
                                      "messages": [{"role": "user", "content": prompt}],
                                      "chat_template_kwargs": {"enable_thinking": False}})
        r = [x for x in call("/metrics?requests=all")["requests"] if x["id"] not in before][0]
        read = r.get("prompt_read") or r["prompt_tokens"]
        print(json.dumps({"tag": tag, "prompt_tokens": r["prompt_tokens"], "read": read,
                          "prompt_ms": r["prompt_ms"],
                          "prefill_tok_s": round(read / r["prompt_ms"] * 1000)}), flush=True)


if __name__ == "__main__":
    main()
