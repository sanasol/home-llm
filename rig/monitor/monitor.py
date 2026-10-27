"""Rig monitor: one page for the whole box — Strata main (:8001) + two HyperQwen
engines (:18020 A speed, :18021 B 262K), all three GPUs and the host.

Samples Strata /metrics (JSON), both vLLM /metrics (Prometheus), nvidia-smi (+ dmon
-s t for real PCIe Rx/Tx MB/s), /proc, hwmon and the journald access logs of both
HyperQwen units. Every point is persisted to SQLite (hist.db): 1s resolution kept
for 24h, 1-minute rollups for 7 days; finished request rows kept for 7 days too.
Restarting the monitor backfills the live charts from the DB.

API:  /api/state   live gauges + history ring (RAM) + recent requests
      /api/history?range=1h|24h|7d | ?from=T&to=T   persisted points downsampled
      /api/summary?range=... | ?from=T&to=T         per-engine totals over a window
      /, /hw  pages (index.html / hw.html read per request)

Env: STRATA_URL, STRATA_KEY (JSON with api_key), A_URL, B_URL, VLLM_ENV (KEY=VAL
file with VLLM_API_KEY, shared by both HyperQwen), A_UNIT, B_UNIT, HIST_DB, PORT.
"""
import json, os, re, sqlite3, subprocess, threading, time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

STRATA = os.environ.get("STRATA_URL", "http://127.0.0.1:8001")
STRATA_KEY_FILE = os.path.expanduser(os.environ.get("STRATA_KEY", "~/.config/strata/launch-presets/main.json"))
A_URL = os.environ.get("A_URL", "http://127.0.0.1:18020")
B_URL = os.environ.get("B_URL", "http://127.0.0.1:18021")
VLLM_ENV = os.environ.get("VLLM_ENV", "/opt/rig/side.env")
A_UNIT = os.environ.get("A_UNIT", "hyperqwen-a")
B_UNIT = os.environ.get("B_UNIT", "hyperqwen-b")
MAIN_UNIT = os.environ.get("MAIN_UNIT", "llm-main")
PORT = int(os.environ.get("PORT", "8090"))
HIST_DB = os.environ.get("HIST_DB", str(Path(__file__).with_name("hist.db")))
HISTORY = 900
ROLL_S = 60           # rollup bucket
KEEP1 = 86400         # 1s points: 24h
KEEP60 = 30 * 86400   # rollups + request rows: 30 days
PAGE = Path(__file__).with_name("index.html")

# engine table: the page and the request rows label by these keys
ENG = {"main": {"label": "Flash-Next", "sub": "Strata :8001 · RTX 5090"},
       "a": {"label": "Qwen 27B A", "sub": "HyperQwen :18020 · 3090 #1 · speed"},
       "b": {"label": "Qwen 27B B", "sub": "HyperQwen :18021 · 3090 #2 · 262K"}}

history = deque(maxlen=HISTORY)
_last_gs = []
state = {"eng": {k: {"stats": {}} for k in ENG}, "gpus": [], "host": {}, "services": {},
         "requests": [], "clients": [], "updated": 0}
lock = threading.Lock()

GPU_FIELDS = ["index", "name", "uuid", "utilization.gpu", "memory.used", "memory.total",
              "temperature.gpu", "power.draw", "power.limit", "fan.speed",
              "pcie.link.gen.current", "pcie.link.width.current"]
METRIC_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?([^}]*)\}?\s+(\S+)$')
ACCESS_RE = re.compile(r'"(?P<method>[A-Z]+) (?P<path>\S+) HTTP/[^"]*"\s+(?P<status>\d{3})\b')
CLIENT_RE = re.compile(r'(?P<ip>[\w.]+):\d+ - ')
CHAT_PATHS = ("/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/completions")

DB_LOCK = threading.Lock()


def db():
    con = sqlite3.connect(HIST_DB, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def db_init():
    with DB_LOCK, db() as con:
        con.executescript(
            "CREATE TABLE IF NOT EXISTS hist1(t REAL PRIMARY KEY, j TEXT);"
            "CREATE TABLE IF NOT EXISTS hist60(t REAL PRIMARY KEY, j TEXT);"
            "CREATE TABLE IF NOT EXISTS hw60(t REAL PRIMARY KEY, j TEXT);"
            "CREATE TABLE IF NOT EXISTS reqs(t REAL, eng TEXT, j TEXT);")


def db_insert(table, t, j):
    with DB_LOCK, db() as con:
        con.execute(f"INSERT OR REPLACE INTO {table}(t,j) VALUES(?,?)", (t, json.dumps(j)))


def db_insert_req(t, eng, j):
    with DB_LOCK, db() as con:
        con.execute("INSERT INTO reqs(t,eng,j) VALUES(?,?,?)", (t, eng, json.dumps(j)))


def db_prune(now):
    with DB_LOCK, db() as con:
        con.execute("DELETE FROM hist1 WHERE t < ?", (now - KEEP1,))
        con.execute("DELETE FROM hist60 WHERE t < ?", (now - KEEP60,))
        con.execute("DELETE FROM reqs WHERE t < ?", (now - KEEP60,))


def db_backfill():
    try:
        with DB_LOCK, db() as con:
            rows = con.execute("SELECT t,j FROM hist1 ORDER BY t DESC LIMIT ?", (HISTORY,)).fetchall()
        for t, j in reversed(rows):
            history.append(json.loads(j))
    except sqlite3.Error:
        pass


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
            out[m.group(1) + ("{" + m.group(2) + "}" if m.group(2) else "")] = float(m.group(3))
    return out


def getp(m, prefix, suffix=""):
    """Sum of values whose bare metric name (no labels) starts with prefix and ends with suffix."""
    v = None
    for k, val in m.items():
        bare = k.split("{")[0]
        if bare.startswith(prefix) and bare.endswith(suffix):
            v = (v or 0) + val
    return v


def getx(m, name):
    """Exact bare metric name (HyperQwen adds *_by_source_total / *_cached_total variants
    that a prefix match would double-count)."""
    v = None
    for k, val in m.items():
        if k.split("{")[0] == name:
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
        for k in GPU_FIELDS:
            if k in ("name", "uuid"):
                continue
            try:
                g[k] = float(g[k])
            except ValueError:
                g[k] = None
        out.append(g)
    n3090 = 0
    for g in out:
        g["label"] = "5090" if "5090" in g["name"] else "3090"
        if g["label"] == "3090":
            n3090 += 1
            g["label"] = f"3090 #{n3090}"
    # real PCIe throughput (NVML counters via dmon); one 1s sample
    try:
        d = subprocess.run(["nvidia-smi", "dmon", "-s", "t", "-c", "1"], capture_output=True, text=True, timeout=5)
        for line in d.stdout.splitlines():
            p = line.split()
            if p and p[0].isdigit() and len(p) in (3, 5):
                off = 2 if len(p) == 5 else 0
                g = next((x for x in out if int(float(x["index"])) == int(p[off])), None)
                if g:
                    g["pcie_rx_mbps"] = float(p[off + 1])
                    g["pcie_tx_mbps"] = float(p[off + 2])
    except Exception:
        pass
    return out


def hwmon_temp(names):
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
            base = f.name[:-len("_input")]
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


# ------------------------------------------------------------- vLLM engines (A and B)
HIST_KEYS = {"n": "vllm:e2e_request_latency_seconds", "prompt": "vllm:request_prompt_tokens",
             "out": "vllm:request_generation_tokens", "prefill": "vllm:request_prefill_time_seconds",
             "decode": "vllm:request_decode_time_seconds", "ttft": "vllm:time_to_first_token"}


class VllmLog:
    """Access rows + histogram attribution for ONE vLLM unit (mirrors the old side_* globals)."""

    def __init__(self, unit):
        self.unit = unit
        self.rows = []
        self.cursors = set()
        self.hist_prev = None
        self.base_ts = 0.0
        self.acc = None
        self.acc_t = 0.0
        self.acc_end = 0.0
        self.hist_t = 0.0

    def access_rows(self, window=300):
        r = subprocess.run(["journalctl", "-u", self.unit, "-b", "-o", "json", "-n", str(window), "--no-pager"],
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
            if cur in self.cursors:
                continue
            self.cursors.add(cur)
            out.append({"ts": int(j["__REALTIME_TIMESTAMP"]) / 1e6, "path": path,
                        "status": int(a.group("status")), "client": c.group("ip"),
                        "prompt": None, "output": None, "dur": None, "ttft": None,
                        "prefill_tps": None, "decode_tps": None, "approx": False, "n": 1,
                        "saved": False})
        out.sort(key=lambda x: x["ts"])
        return out

    def hist(self, m):
        return {tag: (getp(m, name, "_sum"), getp(m, name, "_count")) for tag, name in HIST_KEYS.items()}

    def attribute(self, hist):
        if not hist:
            return
        now = time.time()
        if self.hist_prev is None:
            self.hist_prev, self.base_ts, self.hist_t = hist, now, now
            return
        prev = self.hist_prev

        def delta(tag):
            s1, c1 = hist[tag]
            s0, c0 = prev[tag]
            if None in (s1, c1, s0, c0):
                return 0.0, 0
            return max(0.0, s1 - s0), max(0, int(round(c1 - c0)))

        if hist["n"][1] is not None and prev["n"][1] is not None and hist["n"][1] < prev["n"][1]:
            self.hist_prev, self.base_ts, self.acc = hist, now, None
            self.hist_t = now
            return
        w_start, w_end = self.hist_t, now
        self.hist_t = now

        n_new = delta("n")[1]
        if n_new > 0:
            if self.acc is None:
                self.acc = {tag: [0.0, 0] for tag in HIST_KEYS}
                self.acc_t, self.acc_end = w_start, w_end
            else:
                self.acc_t = min(self.acc_t, w_start)
                self.acc_end = max(self.acc_end, w_end)
            for tag in HIST_KEYS:
                d_s, d_c = delta(tag)
                self.acc[tag][0] += d_s
                self.acc[tag][1] += d_c
        self.hist_prev = hist

        if self.acc is None:
            return
        n = self.acc["n"][1]
        if n <= 0:
            return
        if now - self.acc_end > 900:
            self.acc = None
            return
        pend = sorted((r for r in self.rows if r["prompt"] is None and r["status"] == 200
                       and self.acc_t - 5 <= (r["ts"] or 0) <= self.acc_end + 15), key=lambda r: r["ts"] or 0)
        k = min(n, len(pend))
        if not k:
            return
        s = {tag: self.acc[tag][0] for tag in HIST_KEYS}
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
        for tag in HIST_KEYS:
            self.acc[tag][0] -= s[tag] * k / n
            self.acc[tag][1] -= k
        if self.acc["n"][1] <= 0:
            self.acc = None

    def persist(self, eng):
        for r in self.rows:
            if r["prompt"] is not None and not r["saved"]:
                r["saved"] = True
                db_insert_req(r["ts"], eng, r)

    def clients(self):
        clients = {}
        for row in self.rows:
            cl = clients.setdefault(row["client"], {"n": 0, "last": row["ts"]})
            cl["n"] += 1
            cl["last"] = max(cl["last"], row["ts"])
        cl = [{"client": k, **v} for k, v in clients.items()]
        cl.sort(key=lambda x: -x["last"])
        return cl[:10]


def vllm_derived(m, prev, prev_t):
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
    d["preempts"] = getp(m, "vllm:num_preemptions_total")
    n = getp(m, "vllm:request_success_total")
    tsu, tc = getp(m, "vllm:time_to_first_token", "_sum"), getp(m, "vllm:time_to_first_token", "_count")
    d["ttft"] = round(tsu / tc, 2) if tc and tsu is not None else None
    ps, pc = getp(m, "vllm:request_prompt_tokens", "_sum"), getp(m, "vllm:request_prompt_tokens", "_count")
    d["avg_prompt"] = round(ps / pc) if pc and ps is not None else None
    gs, gc = getp(m, "vllm:request_generation_tokens", "_sum"), getp(m, "vllm:request_generation_tokens", "_count")
    d["avg_out"] = round(gs / gc) if gc and gs is not None else None
    d["requests"] = n
    # counters for the persisted summary (exact names: prefix matching double-counts
    # vllm:prompt_tokens_by_source_total / _cached_total on HyperQwen)
    d["_c"] = {"req": n or 0,
               "prompt": getx(m, "vllm:prompt_tokens_total") or 0,
               "reused": getx(m, "vllm:prompt_tokens_cached_total") or 0,
               "gen": getx(m, "vllm:generation_tokens_total") or 0,
               "pre_s": getp(m, "vllm:request_prefill_time_seconds", "_sum") or 0,
               "dec_s": getp(m, "vllm:request_decode_time_seconds", "_sum") or 0}
    if prev and prev_t:
        dt = time.time() - prev_t
        if dt > 0:
            def delta(name, want=None):
                def val(mm):
                    tot = None
                    for k, x in mm.items():
                        bare, rest = k.split("{", 1) if "{" in k else (k, "")
                        if bare != name or (want and want not in rest):
                            continue
                        tot = (tot or 0) + x
                    return tot
                a, b = val(m), val(prev)
                return round(max(0.0, a - b) / dt, 1) if a is not None and b is not None else None
            # real compute only: prompt_tokens_total also counts cache-hit tokens -> absurd spikes
            d["prefill_tps"] = delta("vllm:prompt_tokens_by_source_total", 'source="local_compute"')
            d["decode_tps"] = delta("vllm:generation_tokens_total")
    return d


_info_cache = {}  # unit -> (t, data); startup facts, refresh is rare


def engine_info(unit):
    hit = _info_cache.get(unit)
    if hit and time.time() - hit[0] < 1800:
        return hit[1]
    r = subprocess.run(["journalctl", "-u", unit, "-b", "-o", "cat", "--no-pager",
                        "-g", "GPU KV cache size"], capture_output=True, text=True, timeout=60)
    out = {}
    for line in r.stdout.splitlines():
        k = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)
        if k:
            out["kv_tokens"] = int(k.group(1).replace(",", ""))
        m = re.search(r"Maximum concurrency for ([\d,]+) tokens", line)
        if m:
            out["max_len"] = int(m.group(1).replace(",", ""))
    if out:
        _info_cache[unit] = (time.time(), out)
    return out or (hit[1] if hit else {})


def service_state(unit):
    return subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True).stdout.strip()


LOGS = {"a": VllmLog(A_UNIT), "b": VllmLog(B_UNIT)}
_main_req_ts = 0.0   # persist Strata rows newer than this


def sample(prev, prev_t, tick):
    global _last_gs, _main_req_ts
    t = time.time()
    main = None
    raws = {}
    try:
        main = http_json(STRATA + "/metrics", key_from_json(STRATA_KEY_FILE), "requests=all")
    except Exception:
        pass
    vkey = key_from_env(VLLM_ENV)
    for eid, url in (("a", A_URL), ("b", B_URL)):
        try:
            raws[eid] = scrape(url, vkey)
        except Exception:
            raws[eid] = None
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

    mlive = (main or {}).get("live", {})
    mr = (main or {}).get("requests", [])
    mtot = (main or {}).get("totals", {})
    ctx_max = float((main or {}).get("engine", {}).get("max_context") or 0)
    held = max([s.get("held_tokens", 0) for s in mlive.get("slots", [])] or [0])

    stats = {"main": {}}
    if main:
        st = {"running": mlive.get("running"), "state": mlive.get("state"),
              "decode_tps": mlive.get("tok_s") if isinstance(mlive.get("tok_s"), (int, float)) else None,
              "prefill_tps": mlive.get("prefill_tok_s_mean"),
              "ctx_pct": round(100 * held / ctx_max, 1) if ctx_max else None,
              "ctx_max": ctx_max or None,
              "prefix_pct": round(100 * mtot.get("reused", 0) / mtot["prompt_tokens"], 1) if mtot.get("prompt_tokens") else None,
              "ttft": None, "requests": mtot.get("requests"),
              "avg_prompt": None, "avg_out": None, "kv_pct": None,
              "_c": {"req": mtot.get("requests", 0), "prompt": mtot.get("prompt_tokens", 0),
                     "reused": mtot.get("reused", 0), "gen": mtot.get("output_tokens", 0),
                     "pre_s": mtot.get("prompt_ms", 0) / 1000, "dec_s": mtot.get("decode_ms", 0) / 1000}}
        stats["main"] = st
    for eid in ("a", "b"):
        d = vllm_derived(raws[eid], prev.get(eid), prev_t) if raws[eid] else {}
        d.update(_info_cache.get(log_unit(eid), (0, {}))[1])  # cached by the info_refresher thread
        stats[eid] = d

    last_pt = history[-1] if history else {}
    def livev(eid, key, hkey):
        v = stats.get(eid, {}).get(key)
        if v is not None:
            return v
        act = stats.get(eid, {}).get("running") or stats.get(eid, {}).get("state") in ("generating", "reading")
        return last_pt.get(hkey) if act else None
    util = [x["utilization.gpu"] for x in g if x.get("utilization.gpu") is not None] if g else []
    temps = [x["temperature.gpu"] for x in g if x.get("temperature.gpu") is not None] if g else []
    point = {"t": round(t, 1),
             "gpu_util": round(sum(util) / len(util), 1) if util else None,
             "gpu_temp": max(temps) if temps else None,
             "gpu_power": round(sum(x["power.draw"] or 0 for x in g), 0) if g else None,
             "vram_gib": round(sum(x["memory.used"] or 0 for x in g) / 1024, 1) if g else None,
             "cpu_pct": c["cpu_pct"], "disk_r_mbps": c["disk_r_mbps"], "disk_w_mbps": c["disk_w_mbps"],
             "main_tps": livev("main", "decode_tps", "main_tps"),
             "main_prefill_tps": stats["main"].get("prefill_tps") or last_pt.get("main_prefill_tps"),
             "main_running": stats["main"].get("running"),
             "main_ctx_pct": stats["main"].get("ctx_pct"),
             "a_tps": livev("a", "decode_tps", "a_tps"), "a_prefill_tps": stats["a"].get("prefill_tps"),
             "a_running": stats["a"].get("running"), "a_waiting": stats["a"].get("waiting"),
             "a_kv_pct": stats["a"].get("kv_pct"),
             "b_tps": livev("b", "decode_tps", "b_tps"), "b_prefill_tps": stats["b"].get("prefill_tps"),
             "b_running": stats["b"].get("running"), "b_waiting": stats["b"].get("waiting"),
             "b_kv_pct": stats["b"].get("kv_pct")}
    point["counters"] = {"m": stats["main"].get("_c"), "a": stats["a"].get("_c"), "b": stats["b"].get("_c")}
    gs = sorted(g, key=lambda x: x.get("index") or 0) if g else _last_gs
    if gs:
        _last_gs = gs
        point["gpu_util_by"] = [x.get("utilization.gpu") for x in gs]
        point["gpu_temp_by"] = [x.get("temperature.gpu") for x in gs]
        point["gpu_power_by"] = [round(x.get("power.draw") or 0, 1) for x in gs]
        point["vram_by"] = [x.get("memory.used") for x in gs]
        point["pcie_rx_by"] = [x.get("pcie_rx_mbps") for x in gs]
        point["pcie_tx_by"] = [x.get("pcie_tx_mbps") for x in gs]

    # request rows
    for eid, log in LOGS.items():
        try:
            log.rows.extend(log.access_rows(window=300))
            if len(log.rows) > 400:
                del log.rows[:-400]
        except Exception:
            pass
        log.attribute(log.hist(raws[eid]) if raws[eid] else None)
        log.persist(eid)
    svc = None
    if tick % 5 == 0:
        svc = {"main": service_state(MAIN_UNIT), "a": service_state(A_UNIT), "b": service_state(B_UNIT)}

    with lock:
        history.append(point)
        state["eng"] = {"main": {"live": main or state["eng"]["main"].get("live"), "stats": stats["main"] or state["eng"]["main"]["stats"]},
                        "a": {"stats": stats["a"] or state["eng"]["a"]["stats"]},
                        "b": {"stats": stats["b"] or state["eng"]["b"]["stats"]}}
        state["host"] = h
        state["hw"] = hw
        state["updated"] = t
        if g:
            state["gpus"] = g
        rows = []
        for r in mr:
            prefill = round(r["prompt_read"] / (r["prompt_ms"] / 1000)) if r.get("prompt_ms") and r.get("prompt_read") else None
            rows.append({"ts": float(r.get("time") or 0), "eng": "main", "status": r.get("finish"),
                         "prompt": r.get("prompt_tokens"), "reused": r.get("reused"),
                         "output": r.get("output_tokens"), "prefill_tps": prefill,
                         "decode_tps": r.get("decode_tok_s"), "dur": r.get("duration_s"),
                         "hit": r.get("hit_rate")})
            if r.get("time") and r["time"] > _main_req_ts:
                _main_req_ts = r["time"]
                db_insert_req(r["time"], "main", {"ts": r["time"], "status": r.get("finish"),
                                                  "prompt": r.get("prompt_tokens"), "reused": r.get("reused"),
                                                  "output": r.get("output_tokens"), "dur": r.get("duration_s"),
                                                  "decode_tps": r.get("decode_tok_s")})
        srows = []
        for eid, log in LOGS.items():
            for r in log.rows:
                if (r["ts"] or 0) >= log.base_ts:
                    srows.append({**r, "eng": eid})
        state["requests"] = sorted(rows + srows, key=lambda r: r["ts"] or 0)  # chronological; page sorts newest-first
        state["clients"] = [{"eng": e, **cl} for e, log in LOGS.items() for cl in log.clients()][:12]
        if svc:
            state["services"] = svc
    db_insert("hist1", t, point)
    if tick % ROLL_S == 0:
        rollup_and_prune()
    return {eid: raws[eid] for eid in ("a", "b")}, t


def rollup_and_prune():
    """Average the last ROLL_S hist1 points into hist60 (counters stay as last-of-window)."""
    now = time.time()
    try:
        with DB_LOCK, db() as con:
            rows = con.execute("SELECT t,j FROM hist1 WHERE t > ? AND t <= ? ORDER BY t",
                               (now - ROLL_S, now)).fetchall()
            if not rows:
                return
            pts = [json.loads(j) for _, j in rows]

            def avg(key):
                vals = [p.get(key) for p in pts if isinstance(p.get(key), (int, float))]
                return round(sum(vals) / len(vals), 2) if vals else None

            def avgby(key):
                cols = [p.get(key) for p in pts if isinstance(p.get(key), list)]
                if not cols:
                    return None
                m = max(len(c) for c in cols)
                out = []
                for i in range(m):
                    vals = [c[i] for c in cols if i < len(c) and isinstance(c[i], (int, float))]
                    out.append(round(sum(vals) / len(vals), 2) if vals else None)
                return out

            last = pts[-1]
            r = {"t": round(now / ROLL_S) * ROLL_S, "n": len(pts),
                 "cpu_pct": avg("cpu_pct"), "disk_r_mbps": avg("disk_r_mbps"), "disk_w_mbps": avg("disk_w_mbps"),
                 "gpu_util": avg("gpu_util"), "gpu_temp": avg("gpu_temp"), "gpu_power": avg("gpu_power"),
                 "vram_gib": avg("vram_gib"),
                 "main_tps": avg("main_tps"), "main_prefill_tps": avg("main_prefill_tps"),
                 "a_tps": avg("a_tps"), "a_prefill_tps": avg("a_prefill_tps"), "a_kv_pct": avg("a_kv_pct"),
                 "b_tps": avg("b_tps"), "b_prefill_tps": avg("b_prefill_tps"), "b_kv_pct": avg("b_kv_pct")}
            for key in ("gpu_util_by", "gpu_temp_by", "gpu_power_by", "vram_by", "pcie_rx_by", "pcie_tx_by"):
                r[key] = avgby(key)
            r["counters"] = last.get("counters")
            con.execute("INSERT OR REPLACE INTO hist60(t,j) VALUES(?,?)", (r["t"], json.dumps(r)))
            # hardware page series: full hwmon snapshot + GPU temps/fans + cpu clock, once a minute
            snap = {"t": r["t"], "hw": state.get("hw", []),
                    "gpus": [{"index": g.get("index"), "label": g.get("label"), "name": g.get("name"),
                              "temperature.gpu": g.get("temperature.gpu"), "fan.speed": g.get("fan.speed"),
                              "power.draw": g.get("power.draw")} for g in state.get("gpus", [])],
                    "host": {k: (state.get("host") or {}).get(k) for k in ("cpu_ghz", "load1", "load5", "cpu_temp")}}
            con.execute("INSERT OR REPLACE INTO hw60(t,j) VALUES(?,?)", (r["t"], json.dumps(snap)))
            con.execute("DELETE FROM hist1 WHERE t < ?", (now - KEEP1,))
            con.execute("DELETE FROM hist60 WHERE t < ?", (now - KEEP60,))
            con.execute("DELETE FROM hw60 WHERE t < ?", (now - KEEP60,))
            con.execute("DELETE FROM reqs WHERE t < ?", (now - KEEP60,))
    except sqlite3.Error as e:
        print("rollup failed:", repr(e), flush=True)


def sampler():
    prev, prev_t, tick = {"a": None, "b": None}, None, 0
    for log in LOGS.values():
        try:
            log.rows.extend(log.access_rows(window=4000))
        except Exception:
            pass
    while True:
        t = time.time()
        try:
            prev, prev_t = sample(prev, prev_t, tick)  # prev_t MUST carry over: it is the delta baseline for live t/s
        except Exception as e:
            print("sample failed:", repr(e), flush=True)
        tick += 1
        time.sleep(max(0.0, 1.0 - (time.time() - t)))


# ------------------------------------------------------------- history/summary APIs
def window_range(qs):
    r = (qs.get("range") or [""])[0]
    now = time.time()
    presets = {"1h": 3600, "6h": 6 * 3600, "24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400, "all": 365 * 86400}
    if r in presets:
        return now - presets[r], now
    try:
        t0 = float((qs.get("from") or [now - 3600])[0])
        t1 = float((qs.get("to") or [now])[0])
        return t0, min(t1, now)
    except ValueError:
        return now - 3600, now


def api_history(qs):
    t0, t1 = window_range(qs)
    span = t1 - t0
    with DB_LOCK, db() as con:
        rows = con.execute("SELECT t,j FROM hist1 WHERE t BETWEEN ? AND ? ORDER BY t", (t0, t1)).fetchall()
        if span > 6 * 3600:  # older than the 1s retention: rollups + the fresh 1s tail
            rows += con.execute("SELECT t,j FROM hist60 WHERE t BETWEEN ? AND ? ORDER BY t", (t0, t1)).fetchall()
    pts = {}
    for t, j in rows:
        pts[round(t, 2)] = json.loads(j)
    pts = [pts[k] for k in sorted(pts)]
    maxn = int((qs.get("max") or [1500])[0])
    if len(pts) > maxn:  # uniform stride downsample
        stride = len(pts) / maxn
        pts = [pts[int(i * stride)] for i in range(maxn)]
    return {"from": t0, "to": t1, "points": pts}


def api_summary(qs):
    t0, t1 = window_range(qs)
    span = t1 - t0
    table = "hist1" if span <= 6 * 3600 else "hist60"
    out = {"from": t0, "to": t1, "engines": {}}
    with DB_LOCK, db() as con:
        rows = con.execute(f"SELECT j FROM {table} WHERE t BETWEEN ? AND ? ORDER BY t", (t0, t1)).fetchall()
        reqs = con.execute("SELECT eng,j FROM reqs WHERE t BETWEEN ? AND ?", (t0, t1)).fetchall()
    prev_c = {"m": None, "a": None, "b": None}
    acc = {e: {"req": 0.0, "prompt": 0.0, "reused": 0.0, "gen": 0.0, "pre_s": 0.0, "dec_s": 0.0} for e in ("m", "a", "b")}
    for (j,) in rows:
        cs = json.loads(j).get("counters") or {}
        for e in ("m", "a", "b"):
            c1 = cs.get(e)
            if not c1:
                continue
            if e != "m" and "reused" not in c1:
                # vLLM points written before the exact-counter fix (prefix sums double-counted
                # *_by_source_total / *_cached_total) — never diff them, just re-baseline
                prev_c[e] = None
                continue
            c0 = prev_c[e]
            if c0:
                for k in acc[e]:
                    d = c1.get(k, 0) - c0.get(k, 0)
                    if d > 0:  # counter reset -> skip this delta
                        acc[e][k] += d
            prev_c[e] = c1
    names = {"m": "main", "a": "a", "b": "b"}
    for e, s in acc.items():
        eng = names[e]
        prompt, reused = s["prompt"], s.get("reused", 0)
        out["engines"][eng] = {
            "requests": round(s["req"]), "prompt": round(prompt), "reused": round(reused),
            "computed": round(prompt - reused), "output": round(s["gen"]),
            "prefill_tps": round((prompt - reused) / s["pre_s"]) if s["pre_s"] > 1 else None,
            "decode_tps": round(s["gen"] / s["dec_s"], 1) if s["dec_s"] > 1 else None}
    rq = {"main": [], "a": [], "b": []}
    for eng, j in reqs:
        try:
            rq[eng].append(json.loads(j))
        except (ValueError, KeyError):
            pass
    for eng, rows_ in rq.items():
        s = out["engines"][eng]
        s["req_rows"] = len(rows_)
        for k in ("prompt", "output", "decode_tps", "dur"):
            vals = [r.get(k) for r in rows_ if isinstance(r.get(k), (int, float))]
            s["avg_" + k] = round(sum(vals) / len(vals), 1) if vals else None
    return out


def api_hw(qs):
    t0, t1 = window_range(qs)
    with DB_LOCK, db() as con:
        rows = con.execute("SELECT t,j FROM hw60 WHERE t BETWEEN ? AND ? ORDER BY t", (t0, t1)).fetchall()
    pts = [json.loads(j) for _, j in rows]
    maxn = int((qs.get("max") or [1500])[0])
    if len(pts) > maxn:
        stride = len(pts) / maxn
        pts = [pts[int(i * stride)] for i in range(maxn)]
    return {"from": t0, "to": t1, "points": pts}


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
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        if u.path == "/api/state":
            with lock:
                m = {k: v for k, v in state.items() if k != "eng"}
                m["eng"] = {eid: {"stats": e["stats"], **({"live": e["live"]} if e.get("live") else {})}
                            for eid, e in state["eng"].items()}
                for eid, e in m["eng"].items():  # main live: the page needs live/engine/cache/totals, not raw metric dumps
                    if "live" in e and isinstance(e["live"], dict):
                        e["live"] = {k: v for k, v in e["live"].items() if k not in ("requests", "history")}
                m["eng_names"] = ENG
                m["history"] = list(history)
                m["requests"] = state["requests"][-200:]
            self.send(json.dumps(m).encode(), "application/json")
        elif u.path == "/api/history":
            self.send(json.dumps(api_history(qs)).encode(), "application/json")
        elif u.path == "/api/summary":
            self.send(json.dumps(api_summary(qs)).encode(), "application/json")
        elif u.path == "/api/hw":
            self.send(json.dumps(api_hw(qs)).encode(), "application/json")
        elif u.path in ("/", "/index.html"):
            self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif u.path in ("/hw", "/hw.html"):
            self.send(Path(__file__).with_name("hw.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_error(404)


def log_unit(eid):
    return A_UNIT if eid == "a" else B_UNIT


def info_refresher():
    while True:
        for unit in (A_UNIT, B_UNIT):
            try:
                engine_info(unit)
            except Exception as e:
                print("engine_info failed:", repr(e), flush=True)
        time.sleep(300)


db_init()
db_backfill()
threading.Thread(target=info_refresher, daemon=True).start()
threading.Thread(target=sampler, daemon=True).start()
ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
