"""NInfer monitor for the Deck: samples NInfer /metrics, nvidia-smi and host memory once a second, keeps 15 minutes
of history, parses finished requests from the ninfer journal, serves a page and JSON on :8090.

Env: NINFER_URL (http://127.0.0.1:8080), NINFER_API_KEY (from /etc/ninfer.env), PORT (8090), UNIT (ninfer).
"""
import json, os, re, subprocess, threading, time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

NINFER = os.environ.get("NINFER_URL", "http://127.0.0.1:8080")
KEY = os.environ.get("NINFER_API_KEY", "")
PORT = int(os.environ.get("PORT", "8090"))
UNIT = os.environ.get("UNIT", "ninfer")
HISTORY = 900
PAGE = Path(__file__).with_name("index.html")  # read per request so page updates don't need a service restart

history = deque(maxlen=HISTORY)
state = {"ninfer": {}, "gpu": {}, "host": {}, "service": "", "requests": [], "kv_capacity_tokens": None,
         "model_load": {"phase": "idle", "pct": None, "loaded_gib": None, "total_gib": None,
                        "speed_mib_s": None, "eta_s": None, "eta_text": ""}, "updated": 0}
lock = threading.Lock()
KV_RE = re.compile(r'capacity \| KV ([\d,]+) tokens')
# Model load: NInfer logs "loading weights | 21.7 GiB", progress lines every ~10s, then
# "weights ready" (finalize: CUDA graphs, host context) and "engine ready".
LOAD_START_RE = re.compile(r'(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+INFO\s+loading weights \| (?P<total>[\d.]+) GiB')
LOAD_PROG_RE = re.compile(
    r'(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+INFO\s+loading weights (?P<pct>[\d.]+)% \| (?P<loaded>[\d.]+) (?P<unit>MiB|GiB)/(?P<total>[\d.]+) GiB \| '
    r'(?P<speed>[\d.]+) MiB/s \| ETA (?:(?P<em>[\d.]+)m )?(?P<es>[\d.]+)s')
LOAD_WEIGHTS_RE = re.compile(r'(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+INFO\s+weights ready')
LOAD_ENGINE_RE = re.compile(r'INFO\s+engine ready')

GPU_FIELDS = ["utilization.gpu", "memory.used", "memory.total", "power.draw", "power.limit", "temperature.gpu",
              "clocks.sm", "fan.speed", "pcie.link.gen.current", "pcie.link.width.current", "name"]
METRIC_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$')
REQ_RE = re.compile(
    r'(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S*\s+INFO\s+req#(?P<id>\d+) done \| (?P<api>[\w-]+) \| (?P<reason>[^|]+?) \| '
    # cache may read "(97.8%, checkpoint)"; durations may read "1.9s" or "172 ms"
    r'prompt (?P<prompt>[\d,]+) \| output (?P<output>[\d,]+) \| cache (?P<cache>[\d,]+) \((?P<cache_pct>[\d.]+)%[^)]*\) \| '
    r'TTFT (?P<ttft>[\d.]+ ?m?s) \| total (?P<total>[\d.]+ ?m?s)(?:.*?prefill (?P<prefill>[\d.]+k?) tok/s)?'
    r'(?:.*?decode (?P<decode>[\d.]+k?) tok/s)?(?:.*?accepted (?P<acc>[\d,]+)/(?P<drafted>[\d,]+))?')


def num(s):
    s = s.replace(",", "").replace(" ", "")
    if s.endswith("ms"):
        return float(s[:-2]) / 1000
    if s.endswith("s"):
        return float(s[:-1])
    if s.endswith("k"):
        return float(s[:-1]) * 1000
    return float(s)


def scrape():
    req = Request(NINFER + "/metrics", headers={"Authorization": f"Bearer {KEY}"} if KEY else {})
    out = {}
    for line in urlopen(req, timeout=3).read().decode().splitlines():
        m = METRIC_RE.match(line)
        if m and "le=" not in (m.group(2) or ""):  # skip histogram buckets
            name = m.group(1) + (m.group(2) or "")
            out[name] = float(m.group(3))
    return out


def gpu():
    r = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=5)
    vals = [v.strip() for v in r.stdout.strip().split(",")]
    if len(vals) != len(GPU_FIELDS):
        return {}
    g = dict(zip(GPU_FIELDS, vals))
    for k in GPU_FIELDS[:-1]:
        try:
            g[k] = float(g[k])
        except ValueError:
            g[k] = None
    return g


def root_device():
    """Block device backing / (e.g. mmcblk0p2 on the microSD) — disk I/O is read for it."""
    try:
        for line in open("/proc/mounts"):
            p = line.split()
            if len(p) >= 2 and p[1] == "/":
                return p[0].removeprefix("/dev/")
    except OSError:
        pass
    return "mmcblk0p2"


def _cpu_stat():
    """(total_jiffies, idle_jiffies) from the aggregate cpu line of /proc/stat."""
    try:
        vals = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    except (OSError, ValueError):
        return None
    return sum(vals), vals[3] + (vals[4] if len(vals) > 4 else 0)


def _disk_sectors(dev):
    """(sectors_read, sectors_written) for a block device from /proc/diskstats."""
    try:
        for line in open("/proc/diskstats"):
            p = line.split()
            if len(p) > 10 and p[2] == dev:
                return int(p[5]), int(p[9])
    except (OSError, ValueError):
        pass
    return None


def disk_usage():
    try:
        s = os.statvfs("/")
        return {"disk_total": s.f_blocks * s.f_frsize, "disk_avail": s.f_bavail * s.f_frsize}
    except OSError:
        return {}


_prev_cpu: tuple[float, int, int] | None = None
_prev_disk: tuple[float, int, int] | None = None
_disk_dev: str | None = None


def cpudisk():
    """CPU utilisation % and root-FS read/write MB/s since the previous tick (None on the first)."""
    global _prev_cpu, _prev_disk, _disk_dev
    out: dict[str, float | None] = {"cpu_pct": None, "disk_r_mbps": None, "disk_w_mbps": None}
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
    if _disk_dev is None:
        _disk_dev = root_device()
    sec = _disk_sectors(_disk_dev)
    if sec:
        sr, sw = sec
        if _prev_disk:
            pt, psr, psw = _prev_disk
            dt = t - pt
            if dt > 0:
                out["disk_r_mbps"] = round(max(0.0, (sr - psr) * 512 / dt / 1048576), 2)
                out["disk_w_mbps"] = round(max(0.0, (sw - psw) * 512 / dt / 1048576), 2)
        _prev_disk = (t, sr, sw)
    return out


def host():
    mem = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        mem[k] = int(v.split()[0]) * 1024
    return {"mem_total": mem["MemTotal"], "mem_available": mem["MemAvailable"],
            "load1": float(open("/proc/loadavg").read().split()[0]), **disk_usage()}


def from_journal():
    """Finished requests (newest first), the KV pool size in tokens, and the current model-load episode."""
    r = subprocess.run(["journalctl", "-u", UNIT, "-o", "cat", "-n", "2000", "--no-pager"],
                       capture_output=True, text=True, timeout=10)
    rows = []
    kv_tokens = None
    ep = None  # the newest load episode: a fresh "loading weights |" line restarts it
    for line in r.stdout.splitlines():
        k = KV_RE.search(line)
        if k:
            kv_tokens = int(k.group(1).replace(",", ""))
        ls = LOAD_START_RE.search(line)
        if ls:
            ep = {"phase": "loading", "pct": 0.0, "loaded_gib": 0.0, "total_gib": float(ls["total"]),
                  "speed_mib_s": None, "eta_s": None, "eta_text": "", "ts": ls["ts"]}
            continue
        lp = LOAD_PROG_RE.search(line)
        if lp:
            if ep is None or ep["phase"] != "loading":
                ep = {"phase": "loading", "pct": float(lp["pct"]),
                      "loaded_gib": float(lp["loaded"]) / (1024 if lp["unit"] == "MiB" else 1),
                      "total_gib": float(lp["total"]), "speed_mib_s": float(lp["speed"]),
                      "eta_s": (float(lp["em"]) * 60 if lp["em"] else 0) + float(lp["es"]),
                      "eta_text": f"{lp['em']}m {lp['es']}s" if lp["em"] else f"{lp['es']}s", "ts": lp["ts"]}
            else:
                ep.update(pct=float(lp["pct"]),
                          loaded_gib=float(lp["loaded"]) / (1024 if lp["unit"] == "MiB" else 1),
                          total_gib=float(lp["total"]), speed_mib_s=float(lp["speed"]),
                          eta_s=(float(lp["em"]) * 60 if lp["em"] else 0) + float(lp["es"]),
                          eta_text=f"{lp['em']}m {lp['es']}s" if lp["em"] else f"{lp['es']}s", ts=lp["ts"])
            continue
        if ep and ep["phase"] == "loading" and LOAD_WEIGHTS_RE.search(line):
            ep.update(phase="finalizing", pct=100.0, eta_s=0.0, eta_text="", speed_mib_s=None, ts=lp["ts"] if lp else ep["ts"])
            continue
        if ep and LOAD_ENGINE_RE.search(line):
            ep.update(phase="idle")  # load finished; nothing left to show
            continue
        m = REQ_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        rows.append({"ts": d["ts"], "id": int(d["id"]), "api": d["api"], "reason": d["reason"],
                     "prompt": int(num(d["prompt"])), "output": int(num(d["output"])), "cache": int(num(d["cache"])),
                     "cache_pct": float(d["cache_pct"]), "ttft": num(d["ttft"]), "total": num(d["total"]),
                     "prefill_tps": num(d["prefill"]) if d["prefill"] else None,
                     "decode_tps": num(d["decode"]) if d["decode"] else None,
                     "accept_pct": round(100 * int(num(d["acc"])) / int(num(d["drafted"])), 1) if d["drafted"] and int(num(d["drafted"])) else None})
    if ep is None or ep["phase"] == "idle":
        ep = ep or {"phase": "idle", "pct": None, "loaded_gib": None, "total_gib": None,
                    "speed_mib_s": None, "eta_s": None, "eta_text": ""}
    else:
        # a still-open episode whose last line is stale (process died mid-load) is not worth showing
        try:
            age = time.time() - datetime.strptime(ep["ts"], "%Y-%m-%d %H:%M:%S.%f").timestamp()
        except (KeyError, ValueError):
            age = float("inf")
        if age > 600:
            ep.update(phase="idle")
    return rows[-50:][::-1], kv_tokens, ep


def service_state():
    return subprocess.run(["systemctl", "is-active", UNIT], capture_output=True, text=True).stdout.strip()


def sample(prev, prev_t, g, tick):
    """One sampling pass; returns the metrics and GPU readings for the next pass."""
    t = time.time()
    try:
        m = scrape()
    except Exception:
        m = {}
    if tick % 2 == 0:
        try:
            g = gpu()
        except Exception:
            g = {}
    h = host()
    c = cpudisk()
    point = {"t": round(t, 1), "prefill_tps": None, "decode_tps": None, "drafted": None, "accepted": None,
             "gpu_util": g.get("utilization.gpu"), "power": g.get("power.draw"), "temp": g.get("temperature.gpu"),
             "vram_gib": g["memory.used"] / 1024 if g.get("memory.used") is not None else None,
             "kv_pct": None, "running": m.get("ninfer_requests_running"), "waiting": m.get("ninfer_requests_waiting"),
             "cpu_pct": c["cpu_pct"], "disk_r_mbps": c["disk_r_mbps"], "disk_w_mbps": c["disk_w_mbps"]}
    kv_used = m.get('ninfer_device_kv_used_page_groups{pool_id="0"}')
    kv_cap = m.get('ninfer_device_kv_capacity_page_groups{pool_id="0"}')
    if kv_used is not None and kv_cap:
        point["kv_pct"] = round(100 * kv_used / kv_cap, 1)
    if prev and m:
        dt = t - prev_t
        d = lambda k: max(0.0, m.get(k, 0) - prev.get(k, 0))
        point["prefill_tps"] = round(d("ninfer_prefill_tokens_total") / dt, 1)
        point["decode_tps"] = round(d("ninfer_generation_tokens_total") / dt, 1)
        # raw counts so the page can smooth the acceptance rate as a ratio of sums
        point["drafted"] = d("ninfer_spec_decode_draft_tokens_total")
        point["accepted"] = d("ninfer_spec_decode_accepted_tokens_total")
    journal = svc = None
    if tick % 5 == 0:  # slow lookups stay outside the lock
        try:
            journal = from_journal()
        except Exception:
            pass
        svc = service_state()
    with lock:
        history.append(point)
        state.update(ninfer=m, gpu=g, host=h, updated=t)
        if journal:
            state["requests"], kv_tokens, load = journal
            state["kv_capacity_tokens"] = kv_tokens or state["kv_capacity_tokens"]
            state["model_load"] = load
        if svc is not None:
            state["service"] = svc
    return (m, t) if m else (None, None), g


def sampler():
    prev, prev_t, g, tick = None, None, {}, 0
    while True:
        t = time.time()
        try:  # never let one bad pass kill the thread: the page would freeze on stale data
            (prev, prev_t), g = sample(prev, prev_t, g, tick)
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
                body = json.dumps({**state, "history": list(history)}).encode()
            self.send(body, "application/json")
        elif self.path in ("/", "/index.html"):
            self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_error(404)


threading.Thread(target=sampler, daemon=True).start()
ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
