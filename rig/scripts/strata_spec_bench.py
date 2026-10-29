#!/usr/bin/env python3
"""A/B bench for Strata (:8001) speculative-decoding settings (docs/21).

Fixed prompt set, temperature 0 (spec decode is lossless, so arms decode
near-identical text and only the drafter's hit rate differs). Per-request decode
t/s + MTP drafts offered/accepted come from Strata /metrics?requests=all.

Groups:
  ru        8 Russian prose prompts, thinking off (<=700 tok)
  ru_think  3 Russian prompts, thinking on (<=1500 tok)
  en        3 English prose prompts
  code      2 agent-style code edits over real files (output repeats the input:
            the case prompt-lookup drafting targets)
  par2      2 Russian prompts sent concurrently (batch slots), x2 rounds

usage: strata_spec_bench.py ARM_TAG [--groups ru,ru_think,en,code,par2] [--out FILE]
"""
import argparse
import json
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8001"
KEY = json.load(open("~/.config/strata/launch-presets/main.json"))["api_key"]
HDR = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
MODEL = "qwen3.8-flash-next-iq3_s"

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
RU_THINK = [
    "Почему небо голубое, а закат красный? Объясни физику явления.",
    "Как правильно настроить резервное копирование домашнего сервера? Дай конкретный план.",
    "Какие риски у квантования языковых моделей до 3 бит и как их уменьшить?",
]
EN = [
    "Explain how speculative decoding works in large language models.",
    "Write a step-by-step guide for replacing thermal paste on a GPU.",
    "Compare PostgreSQL and ClickHouse for analytics workloads.",
]
PAR = [
    ("Расскажи подробно, как устроен двигатель внутреннего сгорания.",
     "Опиши историю развития интернета от ARPANET до наших дней."),
    ("Объясни, как работает протокол TCP и чем он отличается от UDP.",
     "Напиши подробный гид по выбору ноутбука для программиста."),
]
CODE_FILES = [
    ("/opt/rig/monitor/monitor.py", 126, 230),
    ("~/HyperQwen-0.32/prepare/build_draft_vocab.py", 1, 110),
]


def code_prompts():
    out = []
    for path, a, b in CODE_FILES:
        src = "".join(open(path, encoding="utf-8").readlines()[a - 1:b])
        out.append("Rewrite the following Python code in full. Add a one-line English docstring to "
                   "every function that does not have one; change nothing else. Output only the "
                   "code.\n\n```python\n" + src + "```")
    return out


def call(path, body=None, timeout=900):
    req = urllib.request.Request(URL + path, headers=HDR,
                                 data=json.dumps(body).encode() if body else None)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def metrics_rows():
    return call("/metrics?requests=all").get("requests", [])


def wait_idle():
    while call("/metrics").get("live", {}).get("running"):
        time.sleep(2)  # wait out foreign traffic so rows stay attributable


def body_for(prompt, think, max_tokens):
    return {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": think}}


def row_of(group, think, prompt, wall, r, foreign=0):
    off, acc = r.get("drafts_offered") or 0, r.get("drafts_accepted") or 0
    return {"group": group, "think": think, "prompt": prompt[:40].replace("\n", " "),
            "wall_s": round(wall, 2), "out": r.get("output_tokens"),
            "decode_tok_s": r.get("decode_tok_s"), "drafts_offered": off,
            "drafts_accepted": acc, "accept_rate": round(acc / off, 3) if off else None,
            "foreign_rows": foreign}


def run_one(group, prompt, think, max_tokens):
    wait_idle()
    before = {r["id"] for r in metrics_rows()}
    t0 = time.time()
    call("/v1/chat/completions", body_for(prompt, think, max_tokens))
    wall = time.time() - t0
    new = [r for r in metrics_rows() if r["id"] not in before]
    r = max(new, key=lambda x: x["time"]) if new else {}
    row = row_of(group, think, prompt, wall, r, len(new) - 1)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def run_par(prompts, max_tokens=700):
    wait_idle()
    before = {r["id"] for r in metrics_rows()}
    t0 = time.time()
    th = [threading.Thread(target=call,
                           args=("/v1/chat/completions", body_for(p, False, max_tokens)))
          for p in prompts]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    new = [r for r in metrics_rows() if r["id"] not in before]
    rows = [row_of("par2", False, "parallel", wall, r) for r in new]
    agg = sum(r["out"] or 0 for r in rows) / wall
    for r in rows:
        r["aggregate_tok_s"] = round(agg, 1)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    return rows


def summarize(rows):
    summ = {}
    for g in ("ru", "ru_think", "en", "code", "par2"):
        gr = [r for r in rows if r["group"] == g and r["decode_tok_s"]]
        if not gr:
            continue
        off = sum(r["drafts_offered"] for r in gr)
        acc = sum(r["drafts_accepted"] for r in gr)
        toks = sum(r["out"] for r in gr)
        dec_s = sum(r["out"] / r["decode_tok_s"] for r in gr)
        s = {"n": len(gr), "tokens": toks, "decode_tok_s": round(toks / dec_s, 1),
             "accept_rate": round(acc / off, 3) if off else None}
        if g == "par2":
            s["aggregate_tok_s"] = round(sum(r["aggregate_tok_s"] for r in gr) / len(gr), 1)
        summ[g] = s
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--groups", default="ru,ru_think,en,code,par2")
    ap.add_argument("--out")
    a = ap.parse_args()
    groups = a.groups.split(",")
    run_one("warmup", EN[0], False, 64)
    rows = []
    if "ru" in groups:
        rows += [run_one("ru", p, False, 700) for p in RU]
    if "ru_think" in groups:
        rows += [run_one("ru_think", p, True, 1500) for p in RU_THINK]
    if "en" in groups:
        rows += [run_one("en", p, False, 700) for p in EN]
    if "code" in groups:
        rows += [run_one("code", p, False, 2500) for p in code_prompts()]
    if "par2" in groups:
        for pair in PAR:
            rows += run_par(pair)
    summ = summarize(rows)
    print("SUMMARY", a.arm, json.dumps(summ))
    out = a.out or f"~/strata_bench_{a.arm}.json"
    json.dump({"arm": a.arm, "rows": rows, "summary": summ}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
