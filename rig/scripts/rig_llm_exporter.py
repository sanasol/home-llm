#!/usr/bin/env python3
"""rig_llm_exporter: uniform Prometheus metrics for ALL three rig LLM engines + GPU/fan labels.

Why: Strata (:8001) speaks custom JSON, so Prometheus alone has no tok/s for flash-next; vLLM
A/B (:18020/:18021) are sampled the same way so one metric family covers all three models.
(29.10: the OpenJet A trial was rolled back — docs/16-openjet-trial.md.)
This exporter samples every engine once per second and publishes one metric family per
quantity with labels {model, engine, gpu, gpu_index, uuid}, so one PromQL query covers every
model/GPU.

Listens 127.0.0.1:9097 (loopback; Prometheus job `rig_llm`). Runs as `al` (journal group,
reads the Strata preset key and /opt/rig/side.env). Stdlib only.

Gauges (live, 1 s sampling):
  rig_llm_up
  rig_llm_decode_tokens_per_second   mean of NON-ZERO 1 s samples over the last 15 s
                                     (= speed while generating; 0 when idle the whole window)
  rig_llm_prefill_tokens_per_second  same, for prompt processing (uncached tokens only)
  rig_llm_requests_running / rig_llm_requests_waiting
  rig_llm_kv_usage_ratio             0..1 (Strata: largest slot ctx / max_context;
                                     vLLM: kv_cache_usage_perc)
  rig_llm_kv_capacity_tokens
Counters (cumulative; resets on engine/exporter restart are handled by rate()):
  rig_llm_requests_total, rig_llm_prompt_tokens_total, rig_llm_prompt_cached_tokens_total,
  rig_llm_generation_tokens_total, rig_llm_prefill_seconds_total, rig_llm_decode_seconds_total
Info joins:
  rig_gpu_info{uuid,index,gpu,role}  = 1  (join nvidia_smi_* on uuid for human names)
  rig_fan_info{chip,sensor,fan,control} = 1  (join node_hwmon_fan_rpm on chip,sensor)
"""
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN = os.environ.get("LISTEN", "127.0.0.1")
PORT = int(os.environ.get("PORT", "9097"))
STRATA = os.environ.get("STRATA_URL", "http://127.0.0.1:8001")
STRATA_KEY_FILE = os.environ.get("STRATA_KEY_FILE", "~/.config/strata/launch-presets/main.json")
A_URL = os.environ.get("A_URL", "http://127.0.0.1:18020")
A_UNIT = os.environ.get("A_UNIT", "hyperqwen-a")
B_UNIT = os.environ.get("B_UNIT", "hyperqwen-b")
B_URL = os.environ.get("B_URL", "http://127.0.0.1:18021")
VLLM_ENV = os.environ.get("VLLM_ENV", "/opt/rig/side.env")
WINDOW = 15  # seconds; matches Prometheus scrape_interval

# engine id -> static labels. gpu_index = nvidia-smi index (0 = 5090, 1 = internal 3090 = B,
# 2 = OCuLink 3090 = A — verified by UUID, see rig-llm skill).
ENGINES = {
    "main": {"model": "qwen3.8-flash-next-iq3_s", "engine": "strata", "gpu": "RTX 5090", "gpu_index": "0",
             "uuid": "5565fcfc-574c-5daa-bd59-e29329f6e466"},
    "a": {"model": "qwen3.8-27b", "engine": "vllm", "gpu": "RTX 3090 #1 (A)", "gpu_index": "2",
          "uuid": "9c7268f4-61e4-a63f-7a6f-e33d57743219"},
    "b": {"model": "qwen3.8-27b", "engine": "vllm", "gpu": "RTX 3090 #2 (B)", "gpu_index": "1",
          "uuid": "2e674bbf-af1e-4ad3-17f6-abc8e47b4f36"},
}
GPUS = {  # uuid -> labels
    "5565fcfc-574c-5daa-bd59-e29329f6e466": {"index": "0", "gpu": "RTX 5090", "role": "flash-next (Strata)"},
    "2e674bbf-af1e-4ad3-17f6-abc8e47b4f36": {"index": "1", "gpu": "RTX 3090 #2 (B)", "role": "27B B (vLLM)"},
    "9c7268f4-61e4-a63f-7a6f-e33d57743219": {"index": "2", "gpu": "RTX 3090 #1 (A)", "role": "27B A (vLLM)"},
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


def read_strata_key():
    try:
        return json.load(open(STRATA_KEY_FILE)).get("api_key")
    except Exception:
        return None


def read_env_key():
    try:
        for line in open(VLLM_ENV):
            if line.startswith("VLLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return None


def get(url, key=None, timeout=3):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


PROM_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)')


def parse_prom(text):
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = PROM_LINE.match(line)
        if m:
            try:
                out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
            except ValueError:
                pass
    return out


def psum(m, name, want=None):
    """Sum a metric across label variants (exact name, optional label substring)."""
    tot = 0.0
    for k, v in m.items():
        base = k.split("{", 1)[0]
        if base == name and (want is None or want in k):
            tot += v
    return tot


# ---------------------------------------------------------------- per-engine samplers
_prev = {}


def rate_from(eid, key, val, t):
    """1 s delta of a monotonically increasing counter; None across resets/first sample."""
    pk = (eid, key)
    pv = _prev.get(pk)
    _prev[pk] = (val, t)
    if pv is None or val < pv[0] or t <= pv[1]:
        return None
    return (val - pv[0]) / (t - pv[1])


def sample_main(t):
    d = json.loads(get(STRATA + "/metrics", read_strata_key()))
    live, tot, eng = d.get("live") or {}, d.get("totals") or {}, d.get("engine") or {}
    ctx_max = float(eng.get("max_context") or 0)
    held = [s.get("held_tokens", 0) or 0 for s in live.get("slots", [])]
    st = live.get("state")
    dec = live.get("tok_s") if isinstance(live.get("tok_s"), (int, float)) else 0.0
    pre = 0.0
    # prompt progress delta while reading; falls back to engine's running mean
    pr = live.get("prompt_read")
    if isinstance(pr, (int, float)):
        r = rate_from("main", "prompt_read", float(pr), t)
        pre = r if r and r > 0 else (live.get("prefill_tok_s_mean") or 0.0)
    else:
        _prev.pop(("main", "prompt_read"), None)
    g = {"running": float(live.get("running") or 0), "waiting": float(live.get("waiting") or live.get("queued") or 0),
         "kv": (max(held) / ctx_max) if (held and ctx_max) else 0.0, "kv_cap": ctx_max}
    c = {"req": tot.get("requests", 0), "prompt": tot.get("prompt_tokens", 0), "cached": tot.get("reused", 0),
         "gen": tot.get("output_tokens", 0), "pre_s": tot.get("prompt_ms", 0) / 1000.0,
         "dec_s": tot.get("decode_ms", 0) / 1000.0}
    return g, c, float(dec or 0) if st != "idle" else 0.0, float(pre or 0)


def sample_vllm(eid, url, t):
    m = parse_prom(get(url + "/metrics", read_env_key()))
    gen = psum(m, "vllm:generation_tokens_total")
    comp = psum(m, "vllm:prompt_tokens_by_source_total", 'source="local_compute"')
    dec = rate_from(eid, "gen", gen, t) or 0.0
    pre = rate_from(eid, "comp", comp, t) or 0.0
    g = {"running": psum(m, "vllm:num_requests_running"), "waiting": psum(m, "vllm:num_requests_waiting"),
         "kv": psum(m, "vllm:kv_cache_usage_perc"), "kv_cap": 0.0}
    c = {"req": psum(m, "vllm:request_success_total"), "prompt": psum(m, "vllm:prompt_tokens_total"),
         "cached": psum(m, "vllm:prompt_tokens_cached_total"), "gen": gen,
         "pre_s": psum(m, "vllm:request_prefill_time_seconds_sum"),
         "dec_s": psum(m, "vllm:request_decode_time_seconds_sum")}
    return g, c, dec, pre


SAMPLERS = {"main": sample_main,
            "a": lambda t: sample_vllm("a", A_URL, t),
            "b": lambda t: sample_vllm("b", B_URL, t)}
UNITS = {"a": A_UNIT, "b": B_UNIT}


def kv_capacity(unit):
    """vLLM KV pool size from the journal startup line (cached; slow grep)."""
    try:
        out = subprocess.run(["journalctl", "-u", unit, "-g", "GPU KV cache size", "-o", "cat",
                              "--no-pager"], capture_output=True, text=True, timeout=60).stdout
        hits = re.findall(r"GPU KV cache size: ([\d,]+) tokens", out)
        return float(hits[-1].replace(",", "")) if hits else 0.0
    except Exception:
        return 0.0


def sampler():
    cap, cap_t = {"a": 0.0, "b": 0.0}, 0.0
    while True:
        t = time.time()
        if t - cap_t > 600:
            for e, u in UNITS.items():
                cap[e] = kv_capacity(u) or cap[e]
            cap_t = t
        for eid, fn in SAMPLERS.items():
            try:
                g, c, dec, pre = fn(t)
                if eid in cap and cap[eid]:
                    g["kv_cap"] = cap[eid]
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
                                 ("kv_cap", "rig_llm_kv_capacity_tokens", "KV pool / context capacity, tokens")):
            fam(name, "gauge", help_, [(E[e], gauges[e][key]) for e in E if key in gauges[e]])
        for key, name, help_ in (("req", "rig_llm_requests_total", "Completed requests"),
                                 ("prompt", "rig_llm_prompt_tokens_total", "Prompt tokens (incl. cached)"),
                                 ("cached", "rig_llm_prompt_cached_tokens_total", "Prompt tokens served from prefix cache"),
                                 ("gen", "rig_llm_generation_tokens_total", "Generated tokens"),
                                 ("pre_s", "rig_llm_prefill_seconds_total", "Time spent in prefill"),
                                 ("dec_s", "rig_llm_decode_seconds_total", "Time spent in decode")):
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
