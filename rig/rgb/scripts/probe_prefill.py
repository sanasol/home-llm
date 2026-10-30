#!/usr/bin/env python3
"""Send a large prompt to a lane and watch Strata live-block + exporter prefill
during the prefill phase. Usage: probe_prefill.py [port] [prompt_kchars]"""
import json, sys, threading, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8003
KCH = int(sys.argv[2]) if len(sys.argv) > 2 else 60
PRESET = {8003: "lane2.json", 8004: "lane3.json", 8001: "main.json"}.get(PORT, "lane2.json")
KEY = json.load(open(f"~/.config/strata/launch-presets/{PRESET}"))["api_key"]

big = "Describe each of these notes in one sentence.\n" + "\n".join(
    f"Note {i}: alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo {i%97}"
    for i in range(KCH * 100))


def fire():
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s",
                       "messages": [{"role": "user", "content": big + "\nSummarize in 30 words."}],
                       "max_tokens": 40, "temperature": 0.7}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    try:
        urllib.request.urlopen(req, timeout=300)
        print(f"request done in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print("ERROR", e, flush=True)


threading.Thread(target=fire, daemon=True).start()
for i in range(25):
    time.sleep(1.0)
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/metrics",
                                 headers={"Authorization": f"Bearer {KEY}"})
    live = (json.loads(urllib.request.urlopen(req, timeout=5).read()) or {}).get("live") or {}
    ex = ""
    try:
        txt = urllib.request.urlopen("http://127.0.0.1:9097/metrics", timeout=1).read().decode()
        for ln in txt.splitlines():
            if "prefill_tokens_per_second" in ln and f'gpu_index="{ {8003:1,8004:2}.get(PORT,0)}"' in ln:
                ex = ln.split()[-1]
    except Exception:
        pass
    print(f"t={i:02d} state={live.get('state')} phase={live.get('phase')} "
          f"prompt_read={live.get('prompt_read')}/{live.get('prompt_total')} "
          f"tok_s={live.get('tok_s')} pre_mean={live.get('prefill_tok_s_mean')} exporter_pre={ex}", flush=True)
