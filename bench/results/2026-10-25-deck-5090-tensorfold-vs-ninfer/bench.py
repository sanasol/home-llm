"""Engine-agnostic OpenAI-API bench: short-prompt decode and long-prompt prefill/decode, greedy, thinking off.

argv: URL MODEL OUT.json [TEXT_DIR]
TEXT_DIR: a directory of source files used as long-prompt filler (default: TensorFold zig sources).
Long prompts start with a unique nonce so no engine can reuse a cached prefix.
"""
import glob, json, os, sys, time, uuid, urllib.request

url, model, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
text_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.expanduser("~/qwen38-27b-vision-rtx-5090/engine/zig/src")
filler = ""
for f in sorted(glob.glob(os.path.join(text_dir, "**/*.zig"), recursive=True)):
    filler += f"\n// ==== {os.path.basename(f)} ====\n" + open(f, errors="ignore").read()


def run(content, max_tokens):
    req = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
           "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
           "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    first = last = None
    chunks = 0
    usage = {}
    r = urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(req).encode(),
                                                      {"Content-Type": "application/json"}), timeout=3600)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("usage"):
            usage = ev["usage"]
        ch = ev.get("choices") or []
        if ch and (ch[0].get("delta") or {}).get("content"):
            now = time.time()
            first = first or now
            last = now
            chunks += 1
    toks = usage.get("completion_tokens") or chunks
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": toks,
            "ttft_s": round(first - t0, 3), "decode_tps": round((toks - 1) / (last - first), 1) if last > first else None,
            "total_s": round(time.time() - t0, 2)}


SHORT = {
    "code": "Write a complete Python module implementing an LRU cache with TTL expiry, type hints, and pytest tests.",
    "prose": "Explain in detail how TCP congestion control works, from slow start to BBR. Write several paragraphs.",
    "list": "List 40 practical tips for speeding up a slow PostgreSQL database, one numbered line each.",
}
res = {"url": url, "model": model, "short": {}, "long": []}
run("Hi", 8)  # warm-up
for name, p in SHORT.items():
    runs = [run(p, 512) for _ in range(3)]
    res["short"][name] = runs
    tps = sorted(x["decode_tps"] for x in runs)
    print(f"short {name}: decode median {tps[1]} tok/s, ttft {runs[1]['ttft_s']} s", flush=True)

for chars in (26000, 105000, 330000):  # ~8K, ~32K, ~100K tokens of Zig source
    body = f"[{uuid.uuid4()}]\n" + filler[:chars] + "\n\nSummarize what the code above does, file by file."
    r = run(body, 256)
    r["chars"] = chars
    res["long"].append(r)
    pt = r["prompt_tokens"] or 0
    print(f"long {pt} tok: prefill {r['ttft_s']} s ({pt / r['ttft_s']:.0f} tok/s), decode {r['decode_tps']} tok/s", flush=True)

json.dump(res, open(out, "w"), indent=1)
