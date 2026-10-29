#!/usr/bin/env python3
"""LiveCodeBench (lite test6, AtCoder medium+hard, stdin) coding bench, reasoning medium; runs ON the rig.
usage: lcb_bench.py PORT TAG STREAMS -> lcb_<TAG>.jsonl ; lcb_bench.py --score lcb_*.jsonl"""
import base64, json, os, pickle, re, statistics, subprocess, sys, tempfile, threading, time, urllib.request, zlib

KEY = open("~/HyperQwen/api_key.txt").read().strip()
HDR = {"Content-Type": "application/json", "Authorization": "Bear" + "er " + KEY}
MAX_TESTS = 20


def problems():
    out = []
    for l in open("~/blendq/lcb_test6.jsonl"):
        r = json.loads(l)
        if r["platform"] != "atcoder" or r["difficulty"] not in os.environ.get("DIFF", "medium,hard").split(","):
            continue
        tests = json.loads(r["public_test_cases"])
        try:
            tests += json.loads(pickle.loads(zlib.decompress(base64.b64decode(r["private_test_cases"]))))
        except Exception:  # noqa: BLE001
            pass
        out.append({"id": r["question_id"], "diff": r["difficulty"], "q": r["question_content"],
                    "tests": [t for t in tests if t.get("testtype") == "stdin"][:MAX_TESTS]})
    return out[:int(os.environ.get("LIMIT", "1000"))]


def extract(txt):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", txt or "", re.S)
    return max(blocks, key=len) if blocks else ""


def judge(code, tests):
    if not code:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    try:
        for t in tests:
            try:
                r = subprocess.run([sys.executable, f.name], input=t["input"], capture_output=True, text=True, timeout=10)
            except subprocess.TimeoutExpired:
                return False
            if r.returncode != 0 or r.stdout.split() != t["output"].split():
                return False
        return True
    finally:
        os.unlink(f.name)


def metrics(port):
    t = urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/metrics", headers=HDR)).read().decode()
    d = {}
    for l in t.splitlines():
        for k in ("accepted", "draft"):
            if l.startswith(f"vllm:spec_decode_num_{k}_tokens_total"):
                d[k] = float(l.split()[-1])
    return d


def run(port, tag, streams):
    todo = problems()
    lock = threading.Lock()
    out = open(f"lcb_{tag}.jsonl", "w")
    m0 = metrics(port)

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                p = todo.pop(0)
            body = {"model": "qwen3.8-27b", "max_tokens": 28000, "reasoning_effort": "medium",
                    "messages": [{"role": "user", "content": p["q"] + "\n\nSolve this in Python 3. Read from stdin, "
                                  "write to stdout. Put the complete program in a single ```python block."}]}
            t0 = time.time()
            try:
                r = json.load(urllib.request.urlopen(urllib.request.Request(
                    f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(), HDR), timeout=3600))
                m = r["choices"][0]["message"]
                rec = {"id": p["id"], "diff": p["diff"], "ok": judge(extract(m.get("content")), p["tests"]),
                       "usage": r["usage"], "finish": r["choices"][0].get("finish_reason"), "wall": time.time() - t0}
            except Exception as e:  # noqa: BLE001
                rec = {"id": p["id"], "diff": p["diff"], "error": repr(e)[:300], "wall": time.time() - t0}
            with lock:
                out.write(json.dumps(rec) + "\n"); out.flush()

    th = [threading.Thread(target=worker) for _ in range(streams)]
    [x.start() for x in th]; [x.join() for x in th]
    m1 = metrics(port)
    out.write(json.dumps({"mtp": {"accepted": m1["accepted"] - m0["accepted"], "draft": m1["draft"] - m0["draft"]}}) + "\n")


def score(paths):
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        sp = [r["mtp"] for r in rows if "mtp" in r]
        ok = [r for r in rows if "usage" in r]
        ct = [r["usage"]["completion_tokens"] for r in ok]
        tps = [r["usage"]["completion_tokens"] / r["wall"] for r in ok]
        by = {d: f"{sum(r['ok'] for r in ok if r['diff']==d)}/{sum(1 for r in ok if r['diff']==d)}" for d in ("medium", "hard")}
        acc = f"{sp[0]['accepted']/sp[0]['draft']:.0%}" if sp and sp[0]["draft"] else "-"
        print(f"{p}: n={len(ok)} err={sum('error' in r for r in rows)} pass={sum(r['ok'] for r in ok)} {by} "
              f"trunc={sum(r['finish'] != 'stop' for r in ok)} sum_out_tok={sum(ct)} med_out_tok={statistics.median(ct):.0f} "
              f"med_tps={statistics.median(tps):.0f} sum_wall_s={sum(r['wall'] for r in ok):.0f} spec_accept={acc}")


if __name__ == "__main__":
    if sys.argv[1] == "--score":
        score(sys.argv[2:])
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
