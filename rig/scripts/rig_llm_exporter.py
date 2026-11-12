#!/usr/bin/env python3
"""rig_llm_exporter: uniform Prometheus metrics for ALL rig LLM engines + GPU/fan labels.

Topology (30.10.2026, docs/23): one model, Qwen3.8-Flash-Next IQ3_S on Strata, three independent
lanes sharing one expert arena in RAM:
  main  = FN 5090    :8001  (LiteLLM model qwen3.8-flash-next-iq3_s)
  l2    = FN 3090 #1 :8003  (pool member of qwen3.8-flash-next-3090)
  l3    = FN 3090 #2 :8004  (pool member of qwen3.8-flash-next-3090)
  ds    = DS        :18080 (Project Maya DeepSeek-V4.1-Flash on all three GPUs, LiteLLM
          deepseek-v4.1-flash-maya; 12.11.2026). The rig runs FN OR DS (rig_mode.sh), so either the
          three FN lanes or ds report rig_llm_up 1. ds has gpu="all GPUs" and no uuid/gpu_index.
Strata speaks custom JSON (/metrics), so Prometheus alone has no tok/s; this exporter samples every
lane once per second and publishes one metric family per quantity with labels
{model, engine, lane, gpu, gpu_index, uuid}, so one PromQL query covers every lane/GPU.
(History: until 30.10 the 3090s ran vLLM 27B A/B on :18020/:18021 — retired.)

Listens 127.0.0.1:9097 (loopback; Prometheus job `rig_llm`). Runs as `al` (reads the Strata preset
key). Stdlib only.

Gauges (live, 1 s sampling):
  rig_llm_up
  rig_llm_decode_tokens_per_second   mean of NON-ZERO 1 s samples over the last 15 s
                                     (= speed while generating; 0 when idle the whole window)
  rig_llm_prefill_tokens_per_second  same, for prompt processing (uncached tokens only)
  rig_llm_requests_running / rig_llm_requests_waiting
  rig_llm_kv_usage_ratio             0..1 (largest slot ctx / max_context)
  rig_llm_kv_capacity_tokens         max_context
Counters (cumulative; resets on engine/exporter restart are handled by rate()):
  rig_llm_requests_total, rig_llm_prompt_tokens_total, rig_llm_prompt_cached_tokens_total,
  rig_llm_generation_tokens_total, rig_llm_prefill_seconds_total, rig_llm_decode_seconds_total,
  rig_llm_drafts_offered_total, rig_llm_drafts_accepted_total (Strata MTP)
Info joins:
  rig_gpu_info{uuid,index,gpu,role}  = 1  (join nvidia_smi_* on uuid for human names)
  rig_fan_info{chip,sensor,fan,control} = 1  (join node_hwmon_fan_rpm on chip,sensor)
"""
import json
import os
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN = os.environ.get("LISTEN", "127.0.0.1")
PORT = int(os.environ.get("PORT", "9097"))
STRATA_KEY_FILE = os.environ.get("STRATA_KEY_FILE", "~/.config/strata/launch-presets/main.json")
WINDOW = 15  # seconds; matches Prometheus scrape_interval

# engine id -> static labels. gpu_index = nvidia-smi index (0 = 5090, 1 = internal 3090,
# 2 = OCuLink 3090 — verified by UUID, see rig-llm skill).
ENGINES = {
    "main": {"model": "qwen3.8-flash-next-iq3_s", "engine": "strata", "lane": "FN 5090", "gpu": "RTX 5090",
             "gpu_index": "0", "uuid": "5565fcfc-574c-5daa-bd59-e29329f6e466"},
    "l2": {"model": "qwen3.8-flash-next-3090", "engine": "strata", "lane": "FN 3090 #1", "gpu": "RTX 3090 #1",
           "gpu_index": "1", "uuid": "2e674bbf-af1e-4ad3-17f6-abc8e47b4f36"},
    "l3": {"model": "qwen3.8-flash-next-3090", "engine": "strata", "lane": "FN 3090 #2", "gpu": "RTX 3090 #2",
           "gpu_index": "2", "uuid": "9c7268f4-61e4-a63f-7a6f-e33d57743219"},
    "ds": {"model": "deepseek-v4.1-flash-maya", "engine": "maya", "lane": "DS", "gpu": "all GPUs",
           "gpu_index": "all", "uuid": "all"},
}
URLS = {"main": os.environ.get("STRATA_URL", "http://127.0.0.1:8001"),
        "l2": os.environ.get("L2_URL", "http://127.0.0.1:8003"),
        "l3": os.environ.get("L3_URL", "http://127.0.0.1:8004"),
        "ds": os.environ.get("DS_URL", "http://127.0.0.1:18080")}
GPUS = {  # uuid -> labels
    "5565fcfc-574c-5daa-bd59-e29329f6e466": {"index": "0", "gpu": "RTX 5090", "role": "FN 5090 (Strata :8001)"},
    "2e674bbf-af1e-4ad3-17f6-abc8e47b4f36": {"index": "1", "gpu": "RTX 3090 #1", "role": "FN 3090 #1 (Strata :8003)"},
    "9c7268f4-61e4-a63f-7a6f-e33d57743219": {"index": "2", "gpu": "RTX 3090 #2", "role": "FN 3090 #2 (Strata :8004)"},
}
FANS = [  # node_exporter chip/sensor -> human name (map verified by eye 29.10; fan1 = no tach)
    # Bottom 2x P12 Pro intake sit on CHA_FAN4, which nct6799 does not expose (BIOS curve, no tach).
    ("platform_nct6775_656", "fan2", "AIO radiator fans (top exhaust)", "fancontrol (CPU temp)"),
    ("platform_nct6775_656", "fan3", "Rear exhaust 140", "fancontrol (CPU temp)"),
    ("platform_nct6775_656", "fan5", "Front intake 3x140", "fancontrol (CPU temp, above rear)"),
    ("platform_nct6775_656", "fan4", "AIO pump", "BIOS (fixed max)"),
]

lock = threading.Lock()
gauges = {e: {} for e in ENGINES}
counters = {e: {} for e in ENGINES}
up = {e: 0 for e in ENGINES}
rates = {e: {"decode": deque(maxlen=WINDOW), "prefill": deque(maxlen=WINDOW)} for e in ENGINES}
_prev_gen = {}  # eid -> last totals.output_tokens (decode-flow bit between samples)
_ds_drafts = {"off": 0, "acc": 0, "last": time.time(), "seen": time.time()}   # cumulative since exporter start


def read_strata_key():
    try:
        return json.load(open(STRATA_KEY_FILE)).get("api_key")
    except Exception:
        return None


def get(url, key=None, timeout=3):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


# ---------------------------------------------------------------- per-engine sampler
_prev = {}


def rate_from(eid, key, val, t):
    """1 s delta of a monotonically increasing counter; None across resets/first sample."""
    pk = (eid, key)
    pv = _prev.get(pk)
    _prev[pk] = (val, t)
    if pv is None or val < pv[0] or t <= pv[1]:
        return None
    return (val - pv[0]) / (t - pv[1])


def sample_strata(eid, t):
    d = json.loads(get(URLS[eid] + "/metrics", None if eid == "ds" else read_strata_key()))
    live, tot, eng = d.get("live") or {}, d.get("totals") or {}, d.get("engine") or {}
    ctx_max = float(eng.get("max_context") or 0)
    held = [s.get("held_tokens", 0) or 0 for s in live.get("slots", [])]
    st = live.get("state")
    dec = live.get("tok_s") if isinstance(live.get("tok_s"), (int, float)) else 0.0
    pre = 0.0
    # prompt progress delta while reading; falls back to engine's running mean
    pr = live.get("prompt_read")
    if isinstance(pr, (int, float)):
        r = rate_from(eid, "prompt_read", float(pr), t)
        pre = r if r and r > 0 else (live.get("prefill_tok_s_mean") or 0.0)
    else:
        _prev.pop((eid, "prompt_read"), None)
    # single-slot lanes (:8003/:8004) have no `running`/`slots` keys in live —
    # only state/queued/phase. Derive running from state, kv from live token counts.
    run_raw = live.get("running")
    if run_raw is None:
        run_raw = 1 if (st and st != "idle") else 0
    kv = (max(held) / ctx_max) if (held and ctx_max) else 0.0
    if kv == 0.0 and ctx_max and st and st != "idle":
        ctx_used = (live.get("prompt_tokens") or 0) + (live.get("generated") or 0)
        kv = min(1.0, ctx_used / ctx_max)
    # decode flow can't be seen from `state` alone: with 2 slots the engine says
    # "reading" while the other slot still generates — watch output_tokens delta
    prev = _prev_gen.get(eid)
    _prev_gen[eid] = tot.get("output_tokens", 0)
    dec_flow = prev is not None and tot.get("output_tokens", 0) > prev
    phase = (1 if st == "reading" else 0) | (2 if (st == "generating" or dec_flow) else 0)
    g = {"running": float(run_raw or 0), "waiting": float(live.get("waiting") or live.get("queued") or 0),
         "kv": kv, "kv_cap": ctx_max, "phase": float(phase)}
    c = {"req": tot.get("requests", 0), "prompt": tot.get("prompt_tokens", 0), "cached": tot.get("reused", 0),
         "gen": tot.get("output_tokens", 0), "pre_s": tot.get("prompt_ms", 0) / 1000.0,
         "dec_s": tot.get("decode_ms", 0) / 1000.0,
         "drafts_off": tot.get("drafts_offered", 0), "drafts_acc": tot.get("drafts_accepted", 0)}
    if eid == "ds":   # Maya keeps drafts per request only: add each request row once (by its start time)
        for r in d.get("requests") or []:
            rt = float(r.get("time") or 0)
            if rt > _ds_drafts["last"]:
                _ds_drafts["off"] += r.get("drafts_offered") or 0
                _ds_drafts["acc"] += r.get("drafts_accepted") or 0
                _ds_drafts["seen"] = max(_ds_drafts["seen"], rt)
        _ds_drafts["last"] = _ds_drafts["seen"]
        c["drafts_off"], c["drafts_acc"] = _ds_drafts["off"], _ds_drafts["acc"]
    dec_delta = 0.0 if prev is None else max(0.0, tot.get("output_tokens", 0) - prev)
    return g, c, (max(float(dec or 0), dec_delta) if st != "idle" else 0.0), float(pre or 0)


def sampler():
    while True:
        t = time.time()
        for eid in ENGINES:
            try:
                g, c, dec, pre = sample_strata(eid, t)
                ok = 1
            except Exception:
                g, c, dec, pre, ok = None, None, 0.0, 0.0, 0
            with lock:
                up[eid] = ok
                if ok:
                    gauges[eid], counters[eid] = g, c
                rates[eid]["decode"].append(max(0.0, dec))
                rates[eid]["prefill"].append(max(0.0, pre))
        time.sleep(max(0.05, 1.0 - (time.time() - t)))


def active_mean(dq):
    nz = [x for x in dq if x > 0.5]
    return sum(nz) / len(nz) if nz else 0.0


def lbl(d):
    return "{" + ",".join('%s="%s"' % (k, str(v).replace('"', "'")) for k, v in d.items()) + "}"


def render():
    out = []

    def fam(name, typ, help_, rows):
        out.append("# HELP %s %s" % (name, help_))
        out.append("# TYPE %s %s" % (name, typ))
        for labels, v in rows:
            out.append("%s%s %s" % (name, lbl(labels), repr(float(v))))

    with lock:
        E = ENGINES
        fam("rig_llm_up", "gauge", "Engine answered the last 1 s sample", [(E[e], up[e]) for e in E])
        fam("rig_llm_decode_tokens_per_second", "gauge",
            "Decode tok/s: mean of non-zero 1 s samples in the last 15 s (0 = idle)",
            [(E[e], active_mean(rates[e]["decode"])) for e in E])
        fam("rig_llm_prefill_tokens_per_second", "gauge",
            "Prefill (uncached prompt) tok/s: mean of non-zero 1 s samples in the last 15 s",
            [(E[e], active_mean(rates[e]["prefill"])) for e in E])
        for key, name, help_ in (("running", "rig_llm_requests_running", "Requests in progress"),
                                 ("waiting", "rig_llm_requests_waiting", "Requests queued"),
                                 ("kv", "rig_llm_kv_usage_ratio", "KV cache / context fill 0..1"),
                                 ("kv_cap", "rig_llm_kv_capacity_tokens", "KV pool / context capacity, tokens"),
                                 ("phase", "rig_llm_phase", "Engine phase bitmask: 1 prefill (reading), 2 decode (generating/output flowing)")):
            fam(name, "gauge", help_, [(E[e], gauges[e][key]) for e in E if key in gauges[e]])
        for key, name, help_ in (("req", "rig_llm_requests_total", "Completed requests"),
                                 ("prompt", "rig_llm_prompt_tokens_total", "Prompt tokens (incl. cached)"),
                                 ("cached", "rig_llm_prompt_cached_tokens_total", "Prompt tokens served from prefix cache"),
                                 ("gen", "rig_llm_generation_tokens_total", "Generated tokens"),
                                 ("pre_s", "rig_llm_prefill_seconds_total", "Time spent in prefill"),
                                 ("dec_s", "rig_llm_decode_seconds_total", "Time spent in decode"),
                                 ("drafts_off", "rig_llm_drafts_offered_total", "MTP draft tokens offered"),
                                 ("drafts_acc", "rig_llm_drafts_accepted_total", "MTP draft tokens accepted")):
            fam(name, "counter", help_, [(E[e], counters[e][key]) for e in E if key in counters[e]])
    fam("rig_gpu_info", "gauge", "Human labels per GPU uuid (join target)",
        [({"uuid": u, **v}, 1) for u, v in GPUS.items()])
    fam("rig_fan_info", "gauge", "Human labels per board fan header (join target)",
        [({"chip": c, "sensor": s, "fan": f, "control": ctl}, 1) for c, s, f, ctl in FANS])
    return "\n".join(out) + "\n"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/metrics"):
            body = render().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
        else:
            body = b"rig_llm_exporter: see /metrics\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    threading.Thread(target=sampler, daemon=True).start()
    ThreadingHTTPServer((LISTEN, PORT), H).serve_forever()
