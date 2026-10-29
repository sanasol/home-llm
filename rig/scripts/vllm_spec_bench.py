#!/usr/bin/env python3
"""MTP acceptance + decode bench for a HyperQwen vLLM card (:18020 / :18021).

Sequential requests straight to one engine port (bypasses LiteLLM). Every number
comes from the engine's own Prometheus counters, diffed around each request:
  tok/step  = 1 + accepted_tokens / drafts         (MTP acceptance, what the
                                                    HyperQwen docs call tok/step)
  accept    = accepted_tokens / draft_tokens      (per-draft-token rate)
  decode    = generation_tokens / decode_time     (server-side, spec-aware)
A request whose window saw foreign traffic (success counter moved by != 1) is
discarded and retried, so the bench is safe on a card that serves the pool.

  prefill   = local_compute prompt tokens / request_prefill_time (group "prefill":
              fresh nonce-prefixed prompts of ~8/32/64/120K tokens, max_tokens 1)

usage: vllm_spec_bench.py PORT TAG [--temp 0.7] [--groups ru,en,code,prefill] [--n 8]
"""
import argparse
import json
import re
import time
import urllib.request

KEY = open("~/HyperQwen/api_key.txt").read().strip()

RU = [
    "Объясни простыми словами, как работает спекулятивное декодирование в языковых моделях.",
    "Напиши подробную инструкцию, как заменить термопасту на видеокарте.",
    "Расскажи историю Белграда в пяти абзацах.",
    "Сравни PostgreSQL и ClickHouse для аналитики: плюсы, минусы, когда что выбирать.",
    "Составь план тренировок на неделю для начинающего бегуна с пояснениями.",
    "Опиши, как устроена система охлаждения в современном компьютере, и какие бывают ошибки при сборке.",
    "Напиши письмо арендодателю с просьбой продлить договор аренды ещё на год и снизить цену.",
    "Объясни разницу между процессом и потоком в операционной системе, с примерами.",
]
EN = [
    "Explain how speculative decoding works in large language models.",
    "Write a step-by-step guide for replacing thermal paste on a GPU.",
    "Write a short history of Belgrade in five paragraphs.",
    "Compare PostgreSQL and ClickHouse for analytics workloads.",
    "Make a one-week training plan for a beginner runner, with explanations.",
    "Describe how cooling works in a modern PC and common assembly mistakes.",
    "Write a letter to a landlord asking to extend the lease by a year and lower the rent.",
    "Explain the difference between a process and a thread, with examples.",
]
CODE_SRC = ("/opt/rig/monitor/monitor.py", 126, 230)

CTRS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "gen": "vllm:generation_tokens_total",
    "decode_s": "vllm:request_decode_time_seconds_sum",
    "ok": "vllm:request_success_total",
    "prefill_s": "vllm:request_prefill_time_seconds_sum",
}
LOCAL_PROMPT = r'^vllm:prompt_tokens_by_source_total\{[^}]*source="local_compute"[^}]*\} ([0-9.eE+-]+)$'


def scrape(port):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics",
                                 headers={"Authorization": "Bearer " + KEY})
    text = urllib.request.urlopen(req, timeout=10).read().decode()
    out = {}
    for k, name in CTRS.items():
        # sum every label variant of the exact metric name
        out[k] = sum(float(m.group(1)) for m in
                     re.finditer(r"^" + re.escape(name) + r"(?:\{[^}]*\})? ([0-9.eE+-]+)$", text, re.M))
    out["prompt_local"] = sum(float(m.group(1)) for m in re.finditer(LOCAL_PROMPT, text, re.M))
    run = re.search(r"^vllm:num_requests_running(?:\{[^}]*\})? ([0-9.]+)$", text, re.M)
    out["running"] = float(run.group(1)) if run else 0.0
    return out


def chat(port, prompt, temp, max_tokens, think):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp,
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY,
                                          "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def one(port, group, prompt, temp, max_tokens, think=False, tries=6):
    for _ in range(tries):
        while scrape(port)["running"]:
            time.sleep(1)
        a = scrape(port)
        chat(port, prompt, temp, max_tokens, think)
        b = scrape(port)
        d = {k: b[k] - a[k] for k in CTRS}
        if d["ok"] == 1 and d["drafts"] > 0:
            row = {"group": group, "prompt": prompt[:40].replace("\n", " "), "out": int(d["gen"]),
                   "tok_step": round(1 + d["accepted"] / d["drafts"], 3),
                   "accept": round(d["accepted"] / d["draft_tokens"], 3),
                   "decode_tok_s": round(d["gen"] / d["decode_s"], 1) if d["decode_s"] else None,
                   **{k: d[k] for k in ("drafts", "draft_tokens", "accepted", "gen", "decode_s")}}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            return row
        time.sleep(2)  # foreign request overlapped: retry
    print(json.dumps({"group": group, "prompt": prompt[:40], "skipped": "busy"}), flush=True)
    return None


def prefill(port, ktoks):
    """Cold prefill: fresh nonce-prefixed code prompt of ~ktoks K tokens, max_tokens 1."""
    while scrape(port)["running"]:
        time.sleep(1)
    src = open(CODE_SRC[0], encoding="utf-8").read()
    text = (src * (ktoks * 3500 // len(src) + 1))[: ktoks * 3500]
    a = scrape(port)
    chat(port, f"nonce {time.time()}\n```python\n{text}\n```\nSummarize in one line.", 0, 1, False)
    b = scrape(port)
    toks, secs = b["prompt_local"] - a["prompt_local"], b["prefill_s"] - a["prefill_s"]
    row = {"group": "prefill", "prompt_tokens": int(toks), "prefill_s": round(secs, 2),
           "prefill_tok_s": round(toks / secs) if secs else None, "clean": b["ok"] - a["ok"] == 1}
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("tag")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--groups", default="ru,en,code")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    groups = a.groups.split(",")
    one(a.port, "warmup", EN[0], a.temp, 64)
    rows = []
    if "ru" in groups:
        rows += [one(a.port, "ru", p, a.temp, 700) for p in RU[:a.n]]
    if "en" in groups:
        rows += [one(a.port, "en", p, a.temp, 700) for p in EN[:a.n]]
    if "code" in groups:
        path, s, e = CODE_SRC
        src = "".join(open(path, encoding="utf-8").readlines()[s - 1:e])
        p = ("Rewrite the following Python code in full. Add a one-line English docstring to every "
             "function that does not have one; change nothing else. Output only the code.\n\n```python\n"
             + src + "```")
        rows += [one(a.port, "code", p, a.temp, 2500)]
    rows = [r for r in rows if r]
    summ = {}
    if "prefill" in groups:
        pf = [prefill(a.port, k) for k in (8, 32, 64, 120)]
        summ["prefill"] = {str(r["prompt_tokens"]): r["prefill_tok_s"] for r in pf if r["clean"]}
    for g in ("ru", "en", "code"):
        gr = [r for r in rows if r["group"] == g]
        if not gr:
            continue
        S = {k: sum(r[k] for r in gr) for k in ("drafts", "draft_tokens", "accepted", "gen", "decode_s")}
        summ[g] = {"n": len(gr), "tok_step": round(1 + S["accepted"] / S["drafts"], 2),
                   "accept": round(S["accepted"] / S["draft_tokens"], 3),
                   "decode_tok_s": round(S["gen"] / S["decode_s"], 1)}
    print("SUMMARY", a.tag, f"port={a.port} temp={a.temp}", json.dumps(summ))
    json.dump({"tag": a.tag, "port": a.port, "temp": a.temp, "rows": rows, "summary": summ},
              open(f"~/vllm_spec_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
