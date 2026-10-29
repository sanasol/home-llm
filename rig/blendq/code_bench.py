#!/usr/bin/env python3
"""Coding bench (HumanEval, 164 tasks, reasoning medium, T0.7) against one engine; runs ON the rig.
usage: code_bench.py PORT TAG STREAMS   -> code_<TAG>.jsonl ; then code_bench.py --score code_*.jsonl"""
import json, os, re, subprocess, sys, tempfile, threading, time, urllib.request, statistics

KEY = open("~/HyperQwen/api_key.txt").read().strip()


def tasks():
    from datasets import load_dataset
    return list(load_dataset("openai/openai_humaneval", split="test"))


def extract(txt):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", txt or "", re.S)
    return max(blocks, key=len) if blocks else (txt or "")


def check(t, code):
    prog = code + "\n\n" + t["test"] + f"\ncheck({t['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
    try:
        r = subprocess.run([sys.executable, f.name], capture_output=True, timeout=20)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(f.name)


def metrics(port):
    t = urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/metrics",
                               headers={"Authorization": "Bear" + "er " + KEY})).read().decode()
    d = {}
    for l in t.splitlines():
        for k in ("accepted", "draft"):
            if l.startswith(f"vllm:spec_decode_num_{k}_tokens_total"):
                d[k] = float(l.split()[-1])
    return d


def run(port, tag, streams):
    todo = list(tasks())
    lock = threading.Lock()
    out = open(f"code_{tag}.jsonl", "w")
    hdr = {"Content-Type": "application/json", "Authorization": "Bear" + "er " + KEY}
    m0 = metrics(port)

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                t = todo.pop(0)
            body = {"model": "qwen3.8-27b", "max_tokens": 16000, "reasoning_effort": "medium",
                    "messages": [{"role": "user", "content": "Complete this Python function. Reply with the complete "
                                  "function (with imports) in a single ```python block.\n\n" + t["prompt"]}]}
            t0 = time.time()
            try:
                r = json.load(urllib.request.urlopen(urllib.request.Request(
                    f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(), hdr), timeout=1800))
                m = r["choices"][0]["message"]
                content = m.get("content") or ""
                rec = {"id": t["task_id"], "ok": check(t, extract(content)), "usage": r["usage"],
                       "finish": r["choices"][0].get("finish_reason"), "wall": time.time() - t0,
                       "reasoning_chars": len(m.get("reasoning_content") or m.get("reasoning") or "")}
            except Exception as e:  # noqa: BLE001
                rec = {"id": t["task_id"], "error": repr(e)[:300], "wall": time.time() - t0}
            with lock:
                out.write(json.dumps(rec) + "\n"); out.flush()

    th = [threading.Thread(target=worker) for _ in range(streams)]
    [x.start() for x in th]; [x.join() for x in th]
    m1 = metrics(port)
    if m0 and m1:
        a, d = m1["accepted"] - m0["accepted"], m1["draft"] - m0["draft"]
        out.write(json.dumps({"mtp": {"accepted": a, "draft": d}}) + "\n")


def score(paths):
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        sp = [r["mtp"] for r in rows if "mtp" in r]
        ok = [r for r in rows if "usage" in r]
        ct = [r["usage"]["completion_tokens"] for r in ok]
        tps = [r["usage"]["completion_tokens"] / r["wall"] for r in ok]
        acc = f"{sp[0]['accepted']/sp[0]['draft']:.0%} ({sp[0]['accepted']:.0f}/{sp[0]['draft']:.0f})" if sp and sp[0]["draft"] else "-"
        print(f"{p}: n={len(ok)} err={sum('error' in r for r in rows)} pass={sum(r['ok'] for r in ok)}/{len(ok)} "
              f"trunc={sum(r['finish'] != 'stop' for r in ok)} sum_out_tok={sum(ct)} med_out_tok={statistics.median(ct):.0f} "
              f"p90_out_tok={sorted(ct)[int(len(ct)*.9)]} med_tps={statistics.median(tps):.0f} spec_accept={acc}")


if __name__ == "__main__":
    if sys.argv[1] == "--score":
        score(sys.argv[2:])
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
