import json, time, urllib.request, sys
key = open("~/HyperQwen/api_key.txt").read().strip()
port = sys.argv[1] if len(sys.argv) > 1 else "18020"
H = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

def metrics():
    t = urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/metrics", headers=H)).read().decode()
    d = {}
    for l in t.splitlines():
        for k in ("accepted", "draft"):
            if l.startswith(f"vllm:spec_decode_num_{k}_tokens_total"):
                d[k] = float(l.split()[-1])
    return d

def chat(msg, maxt=6000):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": msg}], "max_tokens": maxt, "temperature": 0.7}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(), H)
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=900)); dt = time.time() - t
    m = r["choices"][0]["message"]
    return dt, r["usage"], (m.get("reasoning_content") or m.get("reasoning") or ""), m.get("content") or ""

QS = ["Объясни простыми словами, чем отличается процесс от потока в ОС. 5-6 предложений.",
      "Write a Python function that returns the longest palindromic substring. Code only, brief.",
      "В комнате 3 выключателя и одна лампа в соседней комнате. Как за один заход определить, какой выключатель её включает? Кратко."]
m0 = metrics()
for q in QS:
    dt, u, rc, c = chat(q)
    ct = u["completion_tokens"]
    print(f"--- {dt:.1f}s completion={ct} tps={ct/dt:.0f} reasoning_chars={len(rc)}")
    print(c[:700])
m1 = metrics()
a, d = m1["accepted"] - m0["accepted"], m1["draft"] - m0["draft"]
print(f"MTP accepted/draft = {a:.0f}/{d:.0f} = {a/d:.1%}")
