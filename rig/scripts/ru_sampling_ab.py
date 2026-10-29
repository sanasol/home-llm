#!/usr/bin/env python3
"""Sampling A/B for RU-output glitches (docs/17). Runs ON the rig.

arm 'base': explicit temperature=1.0, min_p=0 (old card defaults)
arm 'new' : no sampling fields -> engine defaults (29.10: Strata 0.7/min_p 0.05, vLLM 0.7)
arm '{...}': explicit JSON sampling fields

usage: ru_sampling_ab.py ENGINE ARM N|A:B STREAMS OUT.jsonl
ENGINE: a | b | main
"""
import json, sys, time, threading, urllib.request, random

ENGINES = {"a": ("http://127.0.0.1:18020", "qwen3.8-27b"), "b": ("http://127.0.0.1:18021", "qwen3.8-27b"),
           "main": ("http://127.0.0.1:8001", "qwen3.8-flash-next-iq3_s")}


def key(engine):
    if engine == "main":
        return json.load(open("~/.config/strata/launch-presets/main.json"))["api_key"]
    return open("~/HyperQwen/api_key.txt").read().strip()


TOPICS = ["настройку WireGuard между тремя серверами", "почему падает скорость генерации LLM на длинном контексте",
          "выбор блока питания для ПК с двумя видеокартами", "резервное копирование PostgreSQL с проверкой восстановления",
          "мониторинг температуры GPU в Grafana", "разницу между prefix cache и KV cache", "как устроен spec decoding (MTP)",
          "организацию воздушного потока в корпусе Corsair 5000D", "квантование моделей: W4A16, IQ3_S, Q4_K_M",
          "настройку systemd-юнита с зависимостями и таймаутами", "диагностику 401 ошибок за обратным прокси",
          "выбор между vLLM и llama.cpp для домашнего сервера", "как работает LiteLLM роутинг между моделями",
          "миграцию сервиса на новый сервер без даунтайма", "покупку электроники в Белграде с доставкой",
          "сравнение DDR5-4800 и DDR5-6000 для инференса", "защиту SSH-доступа на VPS", "работу с git rebase и конфликтами",
          "как читать Prometheus histogram и считать перцентили", "оптимизацию Docker-образа Python-приложения",
          "почему модель иногда отвечает на другом языке", "настройку вентиляторов через fancontrol в Linux",
          "планирование бюджета на облачные GPU против своего железа", "как устроен Mixture-of-Experts",
          "отладку утечки памяти в долгоживущем процессе", "что такое RoPE и почему контекст ограничен",
          "настройку алертов в Telegram из Grafana", "выбор ESP32-платы для голосового ассистента",
          "организацию заметок и базы знаний для команды агентов", "сравнение PCIe x4, x8 и x16 для видеокарт"]
TASKS = ["Объясни подробно, по шагам, с примерами", "Напиши развёрнутый план с рисками и проверками",
         "Сравни варианты в таблице и дай рекомендацию с обоснованием", "Опиши типичные ошибки и как их избежать"]


def prompts(spec):
    """spec 'N' = first N, 'A:B' = slice (vLLM samples are seed-deterministic: give A and B different slices)."""
    rnd = random.Random(42)
    out = [f"{t} {topic}. Ответ — на русском, 400–600 слов." for topic in TOPICS for t in TASKS]
    rnd.shuffle(out)
    a, _, b = spec.partition(":")
    return out[int(a):int(b)] if b else out[:int(a)]


def run(engine, arm, n, streams, out_path):
    url, model = ENGINES[engine]
    k = key(engine)
    hdr = {"Content-Type": "application/json", "Authorization": "Bear" + "er " + k}
    todo = list(enumerate(prompts(n)))
    lock = threading.Lock()
    fout = open(out_path, "a")

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i, p = todo.pop(0)
            body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": 6000,
                    "reasoning_effort": "medium"}
            if arm == "base":
                body.update({"temperature": 1.0, "min_p": 0.0})
            elif arm.startswith("{"):          # explicit params, e.g. '{"temperature":0.6,"top_p":0.8}'
                body.update(json.loads(arm))
            t0 = time.time()
            try:
                req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), hdr)
                r = json.load(urllib.request.urlopen(req, timeout=900))
                m = r["choices"][0]["message"]
                rec = {"engine": engine, "arm": arm, "i": i, "prompt": p, "content": m.get("content") or "",
                       "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
                       "finish": r["choices"][0].get("finish_reason"), "usage": r.get("usage"),
                       "timings": r.get("timings"), "wall": time.time() - t0}
            except Exception as e:  # noqa: BLE001
                rec = {"engine": engine, "arm": arm, "i": i, "error": repr(e)[:300], "wall": time.time() - t0}
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

    ths = [threading.Thread(target=worker) for _ in range(streams)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
