"""Rig monitor: one page for the whole box — Strata main (:8001), vLLM side (:8002), all three GPUs and the host.

Samples Strata /metrics (JSON) and vLLM /metrics (Prometheus) every second, nvidia-smi for all GPUs,
/proc for CPU/RAM/disk, hwmon for temps, and the llm-side journald access log for side requests.
Keeps 15 minutes of history; serves the page + /api/state JSON on :8090. index.html is read per
request, so page-only changes need no service restart.

Env: STRATA_URL, STRATA_KEY (JSON file with "api_key"), VLLM_URL, VLLM_ENV (KEY=VAL file with
VLLM_API_KEY), SIDE_UNIT (llm-side), PORT (8090).
"""
import json, os, re, subprocess, threading, time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

STRATA = os.environ.get("STRATA_URL", "http://127.0.0.1:8001")
STRATA_KEY_FILE = os.path.expanduser(os.environ.get("STRATA_KEY", "~/.config/strata/launch-presets/main.json"))
VLLM = os.environ.get("VLLM_URL", "http://127.0.0.1:8002")
VLLM_ENV = os.environ.get("VLLM_ENV", "/opt/rig/llm-side.env")
SIDE_UNIT = os.environ.get("SIDE_UNIT", "llm-side")
PORT = int(os.environ.get("PORT", "8090"))
HISTORY = 900
PAGE = Path(__file__).with_name("index.html")

history = deque(maxlen=HISTORY)
_last_gs = []
state = {"main": {}, "side": {}, "side_stats": {}, "gpus": [], "host": {}, "services": {},
         "requests": [], "side_requests": [], "clients": [], "updated": 0}
lock = threading.Lock()

GPU_FIELDS = ["index", "name", "uuid", "utilization.gpu", "memory.used", "memory.total",
              "temperature.gpu", "power.draw", "power.limit", "fan.speed",
              "pcie.link.gen.current", "pcie.link.width.current"]
METRIC_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$')
# uvicorn access line from vLLM: INFO:     <tailscale-ip>:49563 - "POST /v1/chat/completions HTTP/1.1" 200 OK
ACCESS_RE = re.compile(r'"(?P<method>[A-Z]+) (?P<path>\S+) HTTP/[^"]*"\s+(?P<status>\d{3})\b')
CLIENT_RE = re.compile(r'(?P<ip>[\w.]+):\d+ - ')
CHAT_PATHS = ("/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/completions")


def key_from_json(path):
    try:
        return json.load(open(path)).get("api_key", "")
    except (OSError, ValueError):
        return ""


def key_from_env(path):
    try:
        for line in open(path):
            k, _, v = line.partition("=")
            if k.strip() == "VLLM_API_KEY":
                return v.strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def http_json(url, key=None, params=None):
    if params:
        url = url + "?" + params
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return json.load(urlopen(Request(url, headers=headers), timeout=3))


def scrape(url, key):
    req = Request(url + "/metrics", headers={"Authorization": f"Bearer {key}"} if key else {})
    out = {}
    for line in urlopen(req, timeout=3).read().decode().splitlines():
        m = METRIC_RE.match(line)
        if m and "le=" not in (m.group(2) or ""):  # skip histogram buckets
            out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


def getp(m, prefix, suffix=""):
    """Sum of values whose bare metric name (no labels) starts with prefix and ends with suffix.
    Sums across label variants (e.g. request_success_total{finished_reason=...})."""
    v = None
    for k, val in m.items():
        bare = k.split("{")[0]
        if bare.startswith(prefix) and bare.endswith(suffix):
            v = (v or 0) + val
    return v


def gpus():
    r = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=5)
    out = []
    for line in r.stdout.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(GPU_FIELDS):
            continue
        g = dict(zip(GPU_FIELDS, vals))
        g["uuid"] = g["uuid"]
        for k in GPU_FIELDS:
            if k in ("name", "uuid"):
                continue
            try:
                g[k] = float(g[k])
            except ValueError:
                g[k] = None
        # short label: model name + which engine owns it (5090 = main, 3090s = side)
        short = "5090" if "5090" in g["name"] else "3090"
        g["label"] = short
        out.append(g)
    n3090 = 0
    for g in out:
        if g["label"] == "3090":
            n3090 += 1
            g["label"] = f"3090 #{n3090}"
    return out


def hwmon_temp(names):
    """First temperature (°C) from an hwmon chip whose name matches one of `names`."""
    try:
        for d in Path("/sys/class/hwmon").iterdir():
            try:
                chip = (d / "name").read_text().strip()
                if chip not in names:
                    continue
                for lab in sorted((d).glob("temp*_input")):
                    v = int(lab.read_text().strip())
                    if v > 0:
                        return round(v / 1000, 1)
            except OSError:
                continue
    except OSError:
        pass
    return None


def root_device():
    try:
        for line in open("/proc/mounts"):
            p = line.split()
            if len(p) >= 2 and p[1] == "/":
                return p[0].removeprefix("/dev/")
    except OSError:
        pass
    return "nvme0n1"


def _cpu_stat():
    try:
        vals = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    except (OSError, ValueError):
        return None
    return sum(vals), vals[3] + (vals[4] if len(vals) > 4 else 0)


def _disk_sectors():
    """(read, write) MiB counters over all nvme* block devices (whole-disk rows only)."""
    r = w = 0
    try:
        for line in open("/proc/diskstats"):
            p = line.split()
            if len(p) > 10 and p[2].startswith("nvme") and "p" not in p[2][4:]:
                r += int(p[5])
                w += int(p[9])
    except (OSError, ValueError):
        return None
    return r, w


_prev_cpu = None
_prev_disk = None


def cpudisk():
    global _prev_cpu, _prev_disk
    out = {"cpu_pct": None, "disk_r_mbps": None, "disk_w_mbps": None}
    t = time.time()
    cpu = _cpu_stat()
    if cpu:
        total, idle = cpu
        if _prev_cpu:
            pt, ptotal, pidle = _prev_cpu
            dt = t - pt
            if dt > 0 and total > ptotal:
                out["cpu_pct"] = round(min(100.0, max(0.0, 100.0 * (1 - (idle - pidle) / (total - ptotal)))), 1)
        _prev_cpu = (t, total, idle)
    sec = _disk_sectors()
    if sec:
        if _prev_disk:
            pt, pr, pw = _prev_disk
            dt = t - pt
            if dt > 0:
                out["disk_r_mbps"] = round(max(0.0, (sec[0] - pr) * 512 / dt / 1048576), 2)
                out["disk_w_mbps"] = round(max(0.0, (sec[1] - pw) * 512 / dt / 1048576), 2)
        _prev_disk = (t, sec[0], sec[1])
    return out


def hwmon_all():
    """Every chip in /sys/class/hwmon: all temperatures (with labels where present) and fans."""
    out = []
    for d in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            chip = (d / "name").read_text().strip()
        except OSError:
            continue
        item = {"chip": chip, "temps": [], "fans": []}
        for f in sorted(d.glob("temp*_input")):
            try:
                v = int(f.read_text())
            except (OSError, ValueError):
                continue
            base = f.name[:-len("_input")]  # temp1_input -> temp1 (label file: temp1_label)
            try:
                label = (d / (base + "_label")).read_text().strip()
            except OSError:
                label = ""
            item["temps"].append({"id": base[4:], "label": label, "c": round(v / 1000, 1)})
        for f in sorted(d.glob("fan*_input")):
            try:
                rpm = int(f.read_text())
            except (OSError, ValueError):
                continue
            item["fans"].append({"id": f.name[:-len("_input")][len("fan"):], "rpm": rpm})
        if item["temps"] or item["fans"]:
            out.append(item)
    return out


def cpu_ghz():
    try:
        for p in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_cur_freq"):
            return round(int(p.read_text().strip()) / 1e6, 2)
    except (OSError, ValueError):
        pass
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("cpu MHz"):
                return round(float(line.split(":")[1]) / 1000, 2)
    except (OSError, ValueError):
        pass
    return None


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def host():
    mem = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        mem[k] = int(v.split()[0]) * 1024
    load = [float(x) for x in open("/proc/loadavg").read().split()[:3]]
    dv = root_device()
    disks = {}
    for mp in ("/", "/models"):
        try:
            s = os.statvfs(mp)
            disks[mp] = {"total": s.f_blocks * s.f_frsize, "avail": s.f_bavail * s.f_frsize}
        except OSError:
            pass
    return {"mem_total": mem["MemTotal"], "mem_available": mem["MemAvailable"],
            "load1": load[0], "load5": load[1], "disks": disks, "disk_dev": dv,
            "cpu_temp": hwmon_temp({"k10temp"}), "nvme_temp": hwmon_temp({"nvme"}), "cpu_ghz": cpu_ghz(),
            "threads": os.cpu_count(), "cpu_name": cpu_name()}


def _access_rows(window=300):
    """New vLLM access rows (chat-ish POSTs) since the last call, tracked by journal cursor.
    Numeric fields are attached later from Prometheus deltas (see attribute)."""
    r = subprocess.run(["journalctl", "-u", SIDE_UNIT, "-b", "-o", "json", "-n", str(window), "--no-pager"],
                       capture_output=True, text=True, timeout=10)
    out = []
    for line in r.stdout.splitlines():
        try:
            j = json.loads(line)
        except ValueError:
            continue
        cur, msg = j.get("__CURSOR"), j.get("MESSAGE")
        if not cur or not isinstance(msg, str):
            continue
        a = ACCESS_RE.search(msg)
        c = CLIENT_RE.search(msg)
        if not a or not c:
            continue
        path = a.group("path").split("?")[0]
        if a.group("method") != "POST" or not path.startswith(CHAT_PATHS):
            continue
        if cur in _side_cursors:
            continue
        _side_cursors.add(cur)
        out.append({"ts": int(j["__REALTIME_TIMESTAMP"]) / 1e6, "path": path,
                    "status": int(a.group("status")), "client": c.group("ip"),
                    "prompt": None, "output": None, "dur": None, "ttft": None,
                    "prefill_tps": None, "decode_tps": None, "approx": False, "n": 1})
    out.sort(key=lambda x: x["ts"])
    return out


# vLLM V1 histogram families: one observation per finished request (sum + count).
HIST_KEYS = {"n": "vllm:e2e_request_latency_seconds", "prompt": "vllm:request_prompt_tokens",
             "out": "vllm:request_generation_tokens", "prefill": "vllm:request_prefill_time_seconds",
             "decode": "vllm:request_decode_time_seconds", "ttft": "vllm:time_to_first_token"}


def side_hist(m):
    return {tag: (getp(m, name, "_sum"), getp(m, name, "_count")) for tag, name in HIST_KEYS.items()}


_side_rows = []
_side_cursors = set()
_side_hist_prev = None
_side_base_ts = 0.0  # rows completed before the histogram baseline can never get numbers
_side_acc = None     # histogram sums of completions not yet handed to access rows
_side_acc_t = 0.0    # first tick the current accumulator window started
_side_acc_end = 0.0  # last tick the accumulator saw new completions
_side_hist_t = 0.0   # timestamp of the previous successful histogram scrape


def attribute(hist):
    """Accumulate histogram sums of finished completions; hand them to un-numbered access rows whose
    journal time falls inside the window those completions finished in. The journal line and the
    counter can surface on different ticks in either order - the accumulator bridges that race.
    Time-window matching prevents rows from a dead vLLM instance (killed orphans, backfilled
    journal) or pre-restart backlog from stealing a live request's numbers.
    One completion in the window -> exact per row; several -> averages marked approx."""
    global _side_hist_prev, _side_base_ts, _side_acc, _side_acc_t, _side_acc_end, _side_hist_t
    if not hist:
        return
    now = time.time()
    if _side_hist_prev is None:
        _side_hist_prev, _side_base_ts, _side_hist_t = hist, now, now
        return
    prev = _side_hist_prev

    def delta(tag):
        s1, c1 = hist[tag]
        s0, c0 = prev[tag]
        if None in (s1, c1, s0, c0):
            return 0.0, 0
        return max(0.0, s1 - s0), max(0, int(round(c1 - c0)))

    if hist["n"][1] is not None and prev["n"][1] is not None and hist["n"][1] < prev["n"][1]:
        # vLLM restarted: counters reset; rows before now are unattributable, take a fresh baseline
        _side_hist_prev, _side_base_ts, _side_acc = hist, now, None
        _side_hist_t = now
        return
    w_start, w_end = _side_hist_t, now
    _side_hist_t = now

    n_new = delta("n")[1]
    if n_new > 0:
        if _side_acc is None:
            _side_acc = {tag: [0.0, 0] for tag in HIST_KEYS}
            _side_acc_t, _side_acc_end = w_start, w_end
        else:
            _side_acc_t = min(_side_acc_t, w_start)
            _side_acc_end = max(_side_acc_end, w_end)
        for tag in HIST_KEYS:
            d_s, d_c = delta(tag)
            _side_acc[tag][0] += d_s
            _side_acc[tag][1] += d_c
    _side_hist_prev = hist

    if _side_acc is None:
        return
    n = _side_acc["n"][1]
    if n <= 0:
        return
    if now - _side_acc_end > 900:
        # the access rows never surfaced: drop the sums so they cannot leak onto a later request
        _side_acc = None
        return
    # match by completion time: the journal line timestamps the completion; rows far outside the
    # window belong to another instance or another era and must never take these numbers
    pend = sorted((r for r in _side_rows if r["prompt"] is None and r["status"] == 200
                   and _side_acc_t - 5 <= (r["ts"] or 0) <= _side_acc_end + 15), key=lambda r: r["ts"] or 0)
    k = min(n, len(pend))
    if not k:
        return
    s = {tag: _side_acc[tag][0] for tag in HIST_KEYS}
    per = lambda tag: s[tag] / n
    for r in pend[:k]:
        r["prompt"] = round(per("prompt"))
        r["output"] = round(per("out"))
        r["dur"] = round(per("n"), 1)
        if s["ttft"]:
            r["ttft"] = round(per("ttft"), 2)
        if s["prefill"] > 0:
            r["prefill_tps"] = round(per("prompt") / per("prefill"))
        if s["decode"] > 0:
            r["decode_tps"] = round(per("out") / per("decode"))
        r["n"] = n
        r["approx"] = n > 1
    for tag in HIST_KEYS:  # consume exactly the k handed-out completions
        _side_acc[tag][0] -= s[tag] * k / n
        _side_acc[tag][1] -= k
    print(f"attr: window_n={n} handed={k} kept={_side_acc['n'][1]}", flush=True)
    if _side_acc["n"][1] <= 0:
        _side_acc = None


def side_clients():
    clients = {}
    for row in _side_rows:
        cl = clients.setdefault(row["client"], {"n": 0, "last": row["ts"]})
        cl["n"] += 1
        cl["last"] = max(cl["last"], row["ts"])
    cl = [{"client": k, **v} for k, v in clients.items()]
    cl.sort(key=lambda x: -x["last"])
    return cl[:10]


def side_info():
    """Side engine facts from the vLLM startup log: KV pool in tokens and max context length."""
    # no -n here: journalctl truncates to N lines BEFORE applying -g, which drops the startup lines
    r = subprocess.run(["journalctl", "-u", SIDE_UNIT, "-b", "-o", "cat", "--no-pager",
                        "-g", "KV cache size|Maximum concurrency"], capture_output=True, text=True, timeout=20)
    out = {}
    for line in r.stdout.splitlines():
        k = re.search(r"KV cache size: ([\d,]+) tokens", line)
        if k:
            out["kv_tokens"] = int(k.group(1).replace(",", ""))
        m = re.search(r"Maximum concurrency for ([\d,]+) tokens", line)
        if m:
            out["max_len"] = int(m.group(1).replace(",", ""))
    return out


def side_derived(m, prev, prev_t):
    """Numbers the page shows for vLLM: gauges, cache/spec stats from counters, tok/s from deltas."""
    d = {}
    d["running"] = getp(m, "vllm:num_requests_running")
    d["waiting"] = getp(m, "vllm:num_requests_waiting")
    kv = getp(m, "vllm:kv_cache_usage_perc")
    d["kv_pct"] = round(100 * kv, 1) if kv is not None else None
    q, h = getp(m, "vllm:prefix_cache_queries_total"), getp(m, "vllm:prefix_cache_hits_total")
    d["prefix_pct"] = round(100 * h / q, 1) if q and h is not None else None
    dq, ac = getp(m, "vllm:spec_decode_num_draft_tokens_total"), getp(m, "vllm:spec_decode_num_accepted_tokens_total")
    d["mtp_pct"] = round(100 * ac / dq, 1) if dq and ac is not None else None
    nst = getp(m, "vllm:spec_decode_num_drafts_total")
    d["mtp_len"] = round(1 + (ac / nst), 2) if nst and ac is not None else None
    d["mtp_acc"] = ac
    d["preempts"] = getp(m, "vllm:num_preemptions_total")
    n = getp(m, "vllm:request_success_total")
    if n is None:
        n = getp(m, "vllm:conversation_completed_responses_total")
    tsu, tc = getp(m, "vllm:time_to_first_token", "_sum"), getp(m, "vllm:time_to_first_token", "_count")
    d["ttft"] = round(tsu / tc, 2) if tc and tsu is not None else None
    ps, pc = getp(m, "vllm:request_prompt_tokens", "_sum"), getp(m, "vllm:request_prompt_tokens", "_count")
    d["avg_prompt"] = round(ps / pc) if pc and ps is not None else None
    gs, gc = getp(m, "vllm:request_generation_tokens", "_sum"), getp(m, "vllm:request_generation_tokens", "_count")
    d["avg_out"] = round(gs / gc) if gc and gs is not None else None
    d["requests"] = n
    if prev and prev_t:
        dt = time.time() - prev_t
        if dt > 0:
            def delta(prefix):
                a, b = getp(m, prefix, "_total"), getp(prev, prefix, "_total")
                return round(max(0.0, a - b) / dt, 1) if a is not None and b is not None else None
            d["prefill_tps"] = delta("vllm:prompt_tokens")
            d["decode_tps"] = delta("vllm:generation_tokens")
    return d


def service_state(unit):
    return subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True).stdout.strip()


def sample(prev, prev_t, tick):
    global _last_gs
    t = time.time()
    main = side = None
    try:
        main = http_json(STRATA + "/metrics", key_from_json(STRATA_KEY_FILE), "requests=all")
    except Exception:
        pass
    try:
        side = scrape(VLLM, key_from_env(VLLM_ENV))
    except Exception:
        pass
    g = []
    if tick % 2 == 0:
        try:
            g = gpus()
        except Exception:
            g = None
        if g is None:
            g = state["gpus"]
    h = host()
    c = cpudisk()
    try:
        hw = hwmon_all()
    except Exception:
        hw = state.get("hw", [])
    sd = side_derived(side, prev, prev_t) if side else {}

    util = [x["utilization.gpu"] for x in g if x.get("utilization.gpu") is not None]
    temps = [x["temperature.gpu"] for x in g if x.get("temperature.gpu") is not None]
    vram_u = sum(x["memory.used"] or 0 for x in g)
    vram_t = sum(x["memory.total"] or 0 for x in g)
    power = sum(x["power.draw"] or 0 for x in g)
    mlive = (main or {}).get("live", {})
    mr = (main or {}).get("requests", [])
    ctx_max = float((main or {}).get("engine", {}).get("max_context") or 0)
    held = max([s.get("held_tokens", 0) for s in mlive.get("slots", [])] or [0])
    last_side = history[-1] if history else {}

    point = {"t": round(t, 1),
             "gpu_util": round(sum(util) / len(util), 1) if util else None,
             "gpu_temp": max(temps) if temps else None,
             "gpu_power": round(power, 0),
             "vram_gib": round(vram_u / 1024, 1),
             "cpu_pct": c["cpu_pct"], "disk_r_mbps": c["disk_r_mbps"], "disk_w_mbps": c["disk_w_mbps"],
             "main_tps": mlive.get("tok_s") if isinstance(mlive.get("tok_s"), (int, float)) else (
                 last_side.get("main_tps") if mlive.get("state") in ("generating", "reading") else None),
             "main_prefill_tps": mlive.get("prefill_tok_s_mean"),
             "main_running": mlive.get("running"),
             "main_ctx_pct": round(100 * held / ctx_max, 2) if ctx_max else None,
             "side_tps": sd.get("decode_tps"), "side_prefill_tps": sd.get("prefill_tps"),
             "side_running": sd.get("running"), "side_waiting": sd.get("waiting"),
             "side_kv_pct": sd.get("kv_pct")}
    gs = sorted(g, key=lambda x: x.get("index") or 0) if g else _last_gs
    if gs:  # per-GPU series so every chart can draw one line per card with a labelled tooltip
        _last_gs = gs
        point["gpu_util_by"] = [x.get("utilization.gpu") for x in gs]
        point["gpu_temp_by"] = [x.get("temperature.gpu") for x in gs]
        point["gpu_power_by"] = [round(x.get("power.draw") or 0, 1) for x in gs]
        point["vram_by"] = [x.get("memory.used") for x in gs]  # MiB
    info = None
    if tick % 5 == 0:
        try:
            info = side_info()
        except Exception:
            pass
    if info:
        state["side_info"] = info
    try:  # new access rows every tick: keeps per-request numbers aligned with histogram deltas
        _side_rows.extend(_access_rows(window=300))
        if len(_side_rows) > 400:
            del _side_rows[:-400]
    except Exception:
        pass
    attribute(side_hist(side) if side else None)
    sd = {**sd, **state.get("side_info", {})} if sd else state.get("side_info", {})
    svc = {"main": service_state("llm-main"), "side": service_state(SIDE_UNIT)} if tick % 5 == 0 else None

    with lock:
        history.append(point)
        state.update(main=main or state["main"], side=side or state["side"], side_stats=sd or state["side_stats"],
                     host=h, hw=hw, updated=t)
        if g:
            state["gpus"] = g
        if main:
            rows = []
            for r in mr:
                prefill = round(r["prompt_read"] / (r["prompt_ms"] / 1000)) if r.get("prompt_ms") and r.get("prompt_read") else None
                rows.append({"ts": float(r.get("time") or 0), "src": "main", "status": r.get("finish"),
                             "prompt": r.get("prompt_tokens"), "reused": r.get("reused"),
                             "output": r.get("output_tokens"), "prefill_tps": prefill,
                             "decode_tps": r.get("decode_tok_s"), "dur": r.get("duration_s"),
                             "hit": r.get("hit_rate")})
            state["requests"] = sorted(rows, key=lambda r: r["ts"] or 0)  # Strata sends newest-first
        state["side_requests"] = [r for r in _side_rows if (r["ts"] or 0) >= _side_base_ts]  # chronological; the page sorts newest-first
        state["clients"] = side_clients()
        if svc:
            state["services"] = svc
    return (side, t) if side else (prev, prev_t)


def sampler():
    prev, prev_t, tick = None, None, 0
    try:  # backfill past rows (numbers not recoverable for them); seed histogram counters from now on
        _side_rows.extend(_access_rows(window=4000))
    except Exception:
        pass
    while True:
        t = time.time()
        try:  # one bad pass must not freeze the page
            prev, prev_t = sample(prev, prev_t, tick)
        except Exception as e:
            print("sample failed:", repr(e), flush=True)
        tick += 1
        time.sleep(max(0.0, 1.0 - (time.time() - t)))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            with lock:
                m = {k: v for k, v in state.items() if k not in ("side", "requests", "side_requests")}
                m["main"] = {k: v for k, v in state["main"].items() if k not in ("requests", "history")}
                m["requests"] = state["requests"][-120:]
                m["side_requests"] = state["side_requests"]
                m["history"] = list(history)
                body = json.dumps(m).encode()
            self.send(body, "application/json")
        elif self.path in ("/", "/index.html"):
            self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path in ("/hw", "/hw.html"):
            self.send(Path(__file__).with_name("hw.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_error(404)


threading.Thread(target=sampler, daemon=True).start()
ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
