"""Rig monitor: one page for the whole box — three Strata lanes of Qwen3.8-Flash-Next
(main :8001 on the RTX 5090, lane2 :8003 on 3090 idx 1, lane3 :8004 on 3090 idx 2;
docs/23), all three GPUs and the host.

Samples every lane's Strata /metrics?requests=all (JSON, one shared API key), nvidia-smi
(+ dmon -s t for real PCIe Rx/Tx MB/s), /proc and hwmon. Every point is persisted to
SQLite (hist.db): 1s resolution kept for 24h, 1-minute rollups for 30 days; finished
request rows kept for 30 days too. Restarting the monitor backfills the live charts.

API:  /api/state   live gauges + history ring (RAM) + recent requests (one merged list, eng field)
      /api/history?range=1h|24h|7d|30d | ?from=T&to=T   persisted points downsampled
      /api/summary?range=... | ?from=T&to=T            per-engine totals over a window
      /api/hw?range=...                                hardware snapshots
      /api/requests?range=...|from/to&eng=&q=&limit=&offset=   persisted request rows, newest first
      /, /hw, /history  pages (index.html / hw.html / history.html read per request)

Request attribution: the LiteLLM affinity hook (rig/litellm/rig_affinity.py) appends one JSONL
line per completion to RIG_REQLOG (session id + conversation key = sha1 of the first user message,
the same key the engine's auto-sessions park under). The monitor tails that log and each lane's
serve log (auto-sessions "restored ... in N ms" lines) and joins both into the request rows:
sess/conv columns and disk_ms = how long that conversation took to come back from disk.

Env: STRATA_URL (main), L2_URL, L3_URL, STRATA_KEY (JSON with api_key, shared by all lanes),
MAIN_UNIT, L2_UNIT, L3_UNIT, HIST_DB, PORT, PROM_URL (Prometheus with the rig:power_* /
rig:electricity_* recording rules, docs/14), LAUNCH_DIR / PRESET_DIR (Strata launch state + presets).
Lane labels, model, context, KV, slots, YaRN and GPU are read from the lanes themselves (/metrics
engine + hardware_static, the unit's --preset, its launch state) — nothing per-config is hardcoded.
Boot progress per lane (/api/state eng.<id>.boot) comes from the unit state + the stage markers the
engine writes into its log on every start.
"""
import json, os, re, sqlite3, subprocess, threading, time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen


STRATA = os.environ.get("STRATA_URL", "http://127.0.0.1:8001")
STRATA_KEY_FILE = os.path.expanduser(os.environ.get("STRATA_KEY", "~/.config/strata/launch-presets/main.json"))
L2_URL = os.environ.get("L2_URL", "http://127.0.0.1:8003")
L3_URL = os.environ.get("L3_URL", "http://127.0.0.1:8004")
MAIN_UNIT = os.environ.get("MAIN_UNIT", "llm-main")
L2_UNIT = os.environ.get("L2_UNIT", "llm-lane2")
L3_UNIT = os.environ.get("L3_UNIT", "llm-lane3")
PORT = int(os.environ.get("PORT", "8090"))
PROM = os.environ.get("PROM_URL", "http://127.0.0.1:9090")
HIST_DB = os.environ.get("HIST_DB", str(Path(__file__).with_name("hist.db")))
HISTORY = 900
ROLL_S = 60           # rollup bucket
KEEP1 = 86400         # 1s points: 24h
KEEP60 = 30 * 86400   # rollups: 30 days
KEEPREQ = 90 * 86400  # request rows: 90 days
PAGE = Path(__file__).with_name("index.html")

# engine table: keys stay fixed (history continuity); labels/subtitles below are only the fallback before the
# first scrape — the live ones come from lane_meta()/lane_names() (GPU, context, KV, slots read from the lanes)
ENG = {"main": {"label": "FN main", "sub": "Strata :8001"},
       "l2": {"label": "FN lane2", "sub": "Strata :8003"},
       "l3": {"label": "FN lane3", "sub": "Strata :8004"}}
LANES = {"main": (STRATA, MAIN_UNIT), "l2": (L2_URL, L2_UNIT), "l3": (L3_URL, L3_UNIT)}
CKEY = {"main": "m", "l2": "l2", "l3": "l3"}   # key in point["counters"] (main stays "m": history continuity)
_ui, _meta, _names, _boot, _up_prev = {}, {}, {}, {}, {}   # per lane: unit info, description, labels, boot progress

history = deque(maxlen=HISTORY)
_last_gs = []
state = {"eng": {k: {"stats": {}} for k in ENG}, "gpus": [], "host": {}, "services": {},
         "requests": [], "updated": 0}
lock = threading.Lock()

GPU_FIELDS = ["index", "name", "uuid", "utilization.gpu", "memory.used", "memory.total",
              "temperature.gpu", "power.draw", "power.limit", "fan.speed",
              "pcie.link.gen.current", "pcie.link.width.current"]


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
        con.execute("DELETE FROM reqs WHERE t < ?", (now - KEEPREQ,))


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


def http_json(url, key=None, params=None):
    if params:
        url = url + "?" + params
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return json.load(urlopen(Request(url, headers=headers), timeout=3))


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


# ------------------------------------------------------------- Strata lanes (main, l2, l3)
def strata_stats(j):
    """Engine stats from one Strata /metrics JSON (same shape for every lane)."""
    live = j.get("live", {}) or {}
    tot = j.get("totals", {}) or {}
    eng = j.get("engine", {}) or {}
    ctx_max = float(eng.get("max_context") or 0)
    held = max([s.get("held_tokens", 0) or 0 for s in live.get("slots", []) or []] or [0])
    off, acc = tot.get("drafts_offered"), tot.get("drafts_accepted")
    return {"running": live.get("running"), "state": live.get("state"), "queued": live.get("queued"),
            "decode_tps": live.get("tok_s") if isinstance(live.get("tok_s"), (int, float)) else None,
            "prefill_tps": live.get("prefill_tok_s_mean") if isinstance(live.get("prefill_tok_s_mean"), (int, float)) else None,
            "ctx_pct": round(100 * held / ctx_max, 1) if ctx_max else None,
            "ctx_max": ctx_max or None,
            "prefix_pct": round(100 * tot.get("reused", 0) / tot["prompt_tokens"], 1) if tot.get("prompt_tokens") else None,
            "mtp_pct": round(100 * acc / off, 1) if off and acc is not None else None,
            "requests": tot.get("requests"),
            "expert_slots": eng.get("expert_slots"), "expert_cache_mib": eng.get("expert_cache_mib"),
            "_c": {"req": tot.get("requests", 0), "prompt": tot.get("prompt_tokens", 0),
                   "reused": tot.get("reused", 0), "gen": tot.get("output_tokens", 0),
                   "pre_s": tot.get("prompt_ms", 0) / 1000, "dec_s": tot.get("decode_ms", 0) / 1000}}


def strata_rows(j, eid):
    """Request rows of one lane, oldest first (Strata lists requests[] NEWEST-first)."""
    rows = []
    for r in sorted((j or {}).get("requests", []) or [], key=lambda x: float(x.get("time") or 0)):
        prefill = round(r["prompt_read"] / (r["prompt_ms"] / 1000)) if r.get("prompt_ms") and r.get("prompt_read") else None
        rows.append({"ts": float(r.get("time") or 0), "eng": eid, "id": r.get("id"), "status": r.get("finish"),
                     "prompt": r.get("prompt_tokens"), "reused": r.get("reused"),
                     "output": r.get("output_tokens"), "prefill_tps": prefill,
                     "decode_tps": r.get("decode_tok_s"), "dur": r.get("duration_s"),
                     "hit": r.get("hit_rate")})
    return rows


def service_state(unit):
    return subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------- lane description + boot progress (all dynamic)
LAUNCH_DIR = Path(os.path.expanduser(os.environ.get("LAUNCH_DIR", "~/.local/state/strata/launch")))
PRESET_DIR = Path(os.path.expanduser(os.environ.get("PRESET_DIR", "~/.config/strata/launch-presets")))


def unit_info(unit):
    """systemd view of a lane unit: state, sub-state, when this run began, and the Strata preset it launches."""
    try:
        out = subprocess.run(["systemctl", "show", "-p", "ActiveState,SubState,InactiveExitTimestamp,ExecStart",
                              "--timestamp=unix", unit], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {}
    d = {}
    for line in out.splitlines():
        k, _, v = line.partition("=")
        d[k] = v
    m = re.search(r"--preset\s+(\S+)", d.get("ExecStart", ""))
    ts = d.get("InactiveExitTimestamp", "").lstrip("@")
    try:
        since = float(ts)
    except ValueError:
        since = None
    return {"active": d.get("ActiveState") or None, "sub": d.get("SubState") or None, "since": since,
            "preset": m.group(1) if m else None}


def _json_file(path):
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return {}
    if isinstance(d, dict):
        d.pop("api_key", None)
        return d
    return {}


def _arg(args, name):
    try:
        i = args.index(name)
        return args[i + 1] if i + 1 < len(args) else None
    except ValueError:
        return None


def ctx_label(n):
    if not n:
        return None
    n = int(n)
    return f"{n // 1048576}M" if n % 1048576 == 0 else f"{round(n / 1024)}K"


def lane_meta(eid, raw, ui, gpu_list):
    """What this lane runs NOW: live engine block first, the last launch's args next, the preset last
    (a preset value that differs from the live one = a change waiting for a restart)."""
    preset = ui.get("preset")
    st = _json_file(LAUNCH_DIR / f"{preset}.json") if preset else {}
    pr = _json_file(PRESET_DIR / f"{preset}.json") if preset else {}
    args = st.get("args") or []
    eng = (raw or {}).get("engine") or {}
    hs = (raw or {}).get("hardware_static") or {}
    live = (raw or {}).get("live") or {}
    gidx = st.get("gpu") if isinstance(st.get("gpu"), int) else ((pr.get("gpus") or [None])[0])
    gname = hs.get("gpu_name") or next((g.get("name") for g in gpu_list or []
                                        if g.get("index") is not None and int(g["index"]) == gidx), None)
    ctx = eng.get("context") or eng.get("max_context") or _arg(args, "--max-context") or pr.get("context")
    kv = eng.get("kv") or _arg(args, "--kv") or pr.get("kv") or "int8"
    slots = eng.get("batch_slots") or live.get("parallel") or pr.get("parallel") or 1
    rope = _arg(args, "--rope-scale") if _arg(args, "--rope-scaling") not in (None, "none") else None
    url = LANES[eid][0]
    port = st.get("port") or pr.get("port") or (int(url.rsplit(":", 1)[1]) if url.rsplit(":", 1)[-1].isdigit() else None)
    pending = []
    if pr and raw:
        if pr.get("context") and ctx and int(pr["context"]) != int(ctx):
            pending.append(f"context {ctx_label(pr['context'])}")
        if (pr.get("kv") or "int8") != kv:
            pending.append(f"KV {pr.get('kv') or 'int8'}")
        if pr.get("parallel") and int(pr["parallel"]) != int(slots):
            pending.append(f"{pr['parallel']} slots")
    return {"model": eng.get("model") or st.get("model_name"), "version": eng.get("version"),
            "ctx": int(ctx) if ctx else None, "ctx_label": ctx_label(ctx), "kv": kv, "slots": int(slots),
            "rope": float(rope) if rope else None, "images": eng.get("images") if raw else pr.get("vision") == "on",
            "gpu_name": gname, "gpu_index": gidx, "port": port, "preset": preset, "unit": LANES[eid][1],
            "log": st.get("log") or str(LAUNCH_DIR / f"{preset}.engine.log"), "pending": pending,
            "arena_mib": eng.get("arena_mib"), "trained_ctx": 262144 if rope else None}


def lane_names(metas):
    """Card labels from the GPU each lane runs on: 'FN 5090', 'FN 3090 #1', 'FN 3090 #2' (numbered by GPU index)."""
    short = {}
    for eid, m in metas.items():
        mm = re.search(r"(\d{4})", m.get("gpu_name") or "")
        short[eid] = mm.group(1) if mm else (m.get("gpu_name") or eid)
    out = {}
    for eid, m in metas.items():
        same = sorted((e for e in metas if short[e] == short[eid]), key=lambda e: metas[e].get("gpu_index") or 0)
        label = f"FN {short[eid]}" + (f" #{same.index(eid) + 1}" if len(same) > 1 else "")
        gpu = (m.get("gpu_name") or "").replace("NVIDIA ", "").replace("GeForce ", "")
        bits = [f":{m.get('port')}", gpu,
                (m.get("ctx_label") or "?") + (f" YaRN×{m['rope']:g}" if m.get("rope") else ""),
                f"KV {m.get('kv')}", f"{m.get('slots')} slot{'s' if m.get('slots', 1) > 1 else ''}"]
        out[eid] = {"label": label, "sub": " · ".join(b for b in bits if b)}
    return out


# engine log markers of one start, in order: (regex, progress %, phase shown on the card)
BOOT_STAGES = [
    (r"engine started", 3, "starting the engine"),
    (r"memory guard:", 6, "checking RAM"),
    (r"MiB of weights loaded", 12, "loading weights"),
    (r"KV streaming: \d+ of", 16, "allocating the KV cache"),
    (r"generate: GPU \d+:", 20, "copying experts into the RAM arena"),   # the long step (~47 GiB)
    (r"expert arena:", 62, "pinning experts for the GPU"),
    (r"generate: loaded [\d.]+ GiB at", 70, "experts loaded"),
    (r"expert cache \d+ slots,", 74, "filling the GPU expert cache"),
    (r"pre-filled \d+ of \d+ slots", 84, "expert cache filled"),
    (r"session is up", 88, "warming up"),
    (r"VRAM free with everything loaded", 93, "capturing CUDA graphs"),
    (r"captured the 1-token window", 100, "ready"),
]
RAM_WAIT = re.compile(r"needs ([\d.]+) GiB of RAM and ([\d.]+) GiB is available; waiting up to (\d+) s.*?(\d+) s so far")
START_TS = re.compile(r"\[strata\] (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) engine started")
_boot_seen = {}   # (eid, start ts, stage index) -> first time the monitor saw that stage (interpolates long stages)


def _log_tail(path, nbytes=262144):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def boot_status(eid, raw, ui, meta, names):
    """None when the lane is up and serving; otherwise {phase, pct, elapsed_s, detail, state} for the card."""
    now = time.time()
    active = ui.get("active")
    text = _log_tail(meta.get("log")) if meta.get("log") else ""
    k = text.rfind("engine started")
    start_ts, stage, detail = None, -1, ""
    if k >= 0:
        block = text[text.rfind("\n", 0, k) + 1:]
        m = START_TS.search(block)
        if m:
            start_ts = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        for i, (rx, _pct, _ph) in enumerate(BOOT_STAGES):
            if re.search(rx, block):
                stage = i
        waits = RAM_WAIT.findall(block)
        if waits and stage <= 1:
            need, avail, cap, so_far = waits[-1]
            detail = f"waiting for RAM: needs {need} GiB, {avail} GiB free ({so_far}/{cap} s)"
    this_run = start_ts is not None and ui.get("since") is not None and start_ts >= ui["since"] - 5
    if raw is not None and (stage == len(BOOT_STAGES) - 1 or not this_run or now - (start_ts or 0) > 1200):
        return None                                     # serving
    if active == "failed":
        return {"state": "failed", "phase": "failed", "pct": 0, "elapsed_s": None,
                "detail": f"unit {meta.get('unit')} failed"}
    if active in ("inactive", "deactivating") and raw is None:
        return {"state": "down", "phase": "stopping" if active == "deactivating" else "stopped", "pct": 0,
                "elapsed_s": None, "detail": f"unit {meta.get('unit')} {active}"}
    if not this_run:                                    # unit (re)started, no engine start in the log yet
        since = ui.get("since")
        if ui.get("sub") == "start-pre":
            phase = f"waiting for {names.get('main', {}).get('label', 'main')}" if eid != "main" else "pre-start"
        elif active == "activating":
            phase = "launcher preparing"
        else:
            phase = "engine not answering"
        return {"state": "starting" if active == "activating" else "down", "phase": phase, "pct": 1,
                "elapsed_s": round(now - since) if since else None, "detail": ""}
    rx, pct, phase = BOOT_STAGES[max(0, stage)]
    key = (eid, start_ts, stage)
    first = _boot_seen.setdefault(key, now)
    if len(_boot_seen) > 200:
        _boot_seen.clear()
    if 0 <= stage < len(BOOT_STAGES) - 1:               # creep toward the next marker while a stage runs
        nxt = BOOT_STAGES[stage + 1][1]
        expect = 14.0 if phase.startswith("copying experts") else 6.0
        pct = pct + (nxt - pct) * min(0.9, (now - first) / expect)
    if detail:
        phase = "waiting for RAM"
    t0 = ui.get("since") or start_ts                  # count from the unit start: launcher prep is part of the wait
    return {"state": "starting", "phase": phase, "pct": round(pct, 1),
            "elapsed_s": round(now - t0) if t0 else None, "detail": detail}


# ------------------------------------------------- request attribution (hook log + serve logs)
REQLOG = Path(os.environ.get("RIG_REQLOG", "/var/lib/litellm/reqlog.jsonl"))
RUNTIME_DIR = Path(os.path.expanduser(os.environ.get("RUNTIME_DIR", "~/.local/state/strata/runtime")))
URL2EID = {url.rstrip("/"): eid for eid, (url, _u) in LANES.items()}
RESTORE_RE = re.compile(r"auto sessions: restored (\d+) tokens from auto-([0-9a-f]{16})\.bin in ([\d.]+) ms")
SAVE_RE = re.compile(r"auto sessions: saved (\d+) tokens \((\d+(?:\.\d+)?) GB\) to auto-([0-9a-f]{16})\.bin in ([\d.]+) ms")
SFAIL_RE = re.compile(r"auto sessions: session file: not enough RAM to read it")
# under --batch the engine-native disk cache does the parking (PR #3341); its log lines carry
# no conversation key, so disk_ms joins to request rows by restored-token count instead
EPARK_RE = re.compile(r"conversation cache: parked (\d+) tokens to disk \((\d+) bytes\) in ([\d.]+) ms")
EBACK_RE = re.compile(r"conversation cache: took back (\d+) tokens from disk \((\d+) bytes\) in ([\d.]+) ms")

_hooks = deque(maxlen=8000)       # attribution events, chronological, each consumed once
_hook_off = [0]                   # byte offset in REQLOG
_disk = {}                        # (eid, conv16) -> (ts, ms) latest disk restore seen
_disk_tok = {}                    # (eid, restored token count) -> (ts, ms): batch-mode, no conv key
_dsum = {e: {"saves": 0, "bytes": 0, "saved_tok": 0, "restores": 0, "rest_tok": 0,
             "rest_ms": 0.0, "fails": 0} for e in ENG}   # cumulative disk-cache activity
_slog = {}                        # eid -> [offset, first_read] of the serve log
_eslog = {}                       # eid -> [offset, first_read] of the engine log
_enrich = {}                      # (eid, engine request id) -> (sess, conv, disk_ms): rows are re-scraped every tick
_conv_lane = {}                   # conv16 -> eid: learned stickiness, for events with no url

def _tail(path, off):
    """Read complete new lines of a growing log; resync when the file shrank."""
    try:
        with open(path, "rb") as f:
            size = f.seek(0, 2)
            if size < off:
                off = max(0, size - 262144)
            f.seek(off)
            data = f.read()
    except OSError:
        return off, []
    nl = data.rfind(b"\n")
    if nl < 0:
        return off, []
    return off + nl + 1, data[:nl].decode("utf-8", "replace").splitlines()

def hook_events():
    """Tail the LiteLLM hook's request log into _hooks: {t, sess, conv, eid|None, used}."""
    _hook_off[0], lines = _tail(REQLOG, _hook_off[0])
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        sid = str(e.get("sid") or "")
        sess, _, digest = sid.partition("|")
        eid = URL2EID.get(str(e.get("url") or "").rstrip("/"))
        conv = str(e.get("conv") or "")
        if eid is None and conv:
            eid = _conv_lane.get(conv)
        if eid is None and str(e.get("model") or "") == "qwen3.8-flash-next-iq3_s":
            eid = "main"
        _hooks.append({"t": float(e.get("t") or 0), "sess": sess or sid, "conv": conv,
                       "digest": digest, "eid": eid, "used": False})

def serve_events(presets):
    """Tail every lane's serve log for auto-sessions disk restores: (eid, conv) -> (ts, ms)."""
    for eid, preset in presets.items():
        if not preset:
            continue
        p = RUNTIME_DIR / ("launch-%s.server.log" % preset)
        st = _slog.setdefault(eid, [0, True])
        if not p.exists():
            continue
        first = st[1]
        st[0], lines = _tail(p, st[0])
        if not lines:
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = time.time()
        for l in lines:
            m = RESTORE_RE.search(l)
            if m:
                # events read on the first pass have no real timestamp: credit the file's mtime
                _disk[(eid, m.group(2))] = (mtime if first else time.time(), float(m.group(3)))
                d = _dsum[eid]
                d["restores"] += 1
                d["rest_tok"] += int(m.group(1))
                d["rest_ms"] += float(m.group(3))
                continue
            m = SAVE_RE.search(l)
            if m:
                d = _dsum[eid]
                d["saves"] += 1
                d["saved_tok"] += int(m.group(1))
                d["bytes"] += int(float(m.group(2)) * (1 << 30))
                continue
            if SFAIL_RE.search(l):
                _dsum[eid]["fails"] += 1
        st[1] = False
    # engine logs: the batch-mode disk cache (PR #3341) parks/restores without a conversation key
    for eid, preset in presets.items():
        if not preset:
            continue
        p = LAUNCH_DIR / ("%s.engine.log" % preset)
        st = _eslog.setdefault(eid, [0, True])
        if not p.exists():
            continue
        first = st[1]
        st[0], lines = _tail(p, st[0])
        if not lines:
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = time.time()
        ts0 = mtime if first else time.time()
        for l in lines:
            m = EPARK_RE.search(l)
            if m:
                d = _dsum[eid]
                d["saves"] += 1
                d["saved_tok"] += int(m.group(1))
                d["bytes"] += int(m.group(2))
                continue
            m = EBACK_RE.search(l)
            if m:
                n, ms = int(m.group(1)), float(m.group(3))
                d = _dsum[eid]
                d["restores"] += 1
                d["rest_tok"] += n
                d["rest_ms"] += ms
                _disk_tok[(eid, n)] = (ts0, ms)
        st[1] = False

def enrich_rows(rows, fresh):
    """Attach session/conversation/disk-restore data to engine request rows.
    fresh: set of (eid, id) newly seen this tick - only those consume hook events;
    everything else is served from the cache (the engine re-lists its recent window every scrape)."""
    for r in rows:
        eid, rid = r["eng"], r.get("id")
        key = (eid, rid)
        if key in _enrich:
            r["sess"], r["conv"], r["disk_ms"] = _enrich[key]
            continue
        if key not in fresh or not rid:
            continue
        # engine "time" is the request START; the hook logs at completion -> join on finish time
        end = r["ts"] + (r.get("dur") or 0)
        lo = r["ts"] - 90
        hit, bestd = None, 1e9
        for h in reversed(_hooks):
            if h["used"] or h["t"] < lo or h["t"] > end + 20:
                continue
            if h["eid"] not in (eid, None):
                continue
            dist = abs(h["t"] - end)
            if dist < bestd:
                hit, bestd = h, dist
        if hit and bestd <= 10:
            hit["used"] = True
            if hit["conv"]:
                _conv_lane.setdefault(hit["conv"], eid)
            d = _disk.pop((eid, hit["conv"]), None)   # one restore belongs to exactly one request
            if d and not (lo <= d[0] <= end + 20):
                d = None
            if d is None:
                # batch mode: the engine disk cache parks without a conversation key,
                # join by restored-token count ~= the request's prompt size
                pt = r.get("prompt") or 0
                for k, ev in list(_disk_tok.items()):
                    if k[0] != eid or not (lo <= ev[0] <= end + 20):
                        continue
                    if pt and abs(k[1] - pt) <= max(1024, 0.05 * pt):
                        d = ev
                        _disk_tok.pop(k)
                        break
            disk_ms = round(d[1], 1) if d else None
            val = (hit["sess"][:60], hit["conv"], disk_ms)
            _enrich[key] = val
            r["sess"], r["conv"], r["disk_ms"] = val
    if len(_enrich) > 20000:
        _enrich.clear()


def _req_seed():
    """Newest persisted request ts per engine, so a monitor restart does not re-insert rows."""
    out = {k: 0.0 for k in ENG}
    try:
        with DB_LOCK, db() as con:
            for eng, t in con.execute("SELECT eng, MAX(t) FROM reqs GROUP BY eng").fetchall():
                if eng in out and t:
                    out[eng] = float(t)
    except sqlite3.Error:
        pass
    return out


_req_ts = {k: 0.0 for k in ENG}   # persist rows newer than this, per engine


def sample(prev, prev_t, tick):
    global _last_gs
    t = time.time()
    key = key_from_json(STRATA_KEY_FILE)
    raws = {}
    for eid, (url, _unit) in LANES.items():
        try:
            raws[eid] = http_json(url + "/metrics", key, "requests=all")
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

    stats = {eid: (strata_stats(raws[eid]) if raws[eid] else {}) for eid in ENG}

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
             "cpu_pct": c["cpu_pct"], "disk_r_mbps": c["disk_r_mbps"], "disk_w_mbps": c["disk_w_mbps"]}
    for eid in ENG:
        point[eid + "_tps"] = livev(eid, "decode_tps", eid + "_tps")
        point[eid + "_prefill_tps"] = livev(eid, "prefill_tps", eid + "_prefill_tps")
        point[eid + "_running"] = stats[eid].get("running")
        point[eid + "_ctx_pct"] = stats[eid].get("ctx_pct")
    point["counters"] = {CKEY[eid]: stats[eid].get("_c") for eid in ENG}
    gs = sorted(g, key=lambda x: x.get("index") or 0) if g else _last_gs
    if gs:
        _last_gs = gs
        point["gpu_util_by"] = [x.get("utilization.gpu") for x in gs]
        point["gpu_temp_by"] = [x.get("temperature.gpu") for x in gs]
        point["gpu_power_by"] = [round(x.get("power.draw") or 0, 1) for x in gs]
        point["vram_by"] = [x.get("memory.used") for x in gs]
        point["pcie_rx_by"] = [x.get("pcie_rx_mbps") for x in gs]
        point["pcie_tx_by"] = [x.get("pcie_tx_mbps") for x in gs]
    # system power = live GPU sum (nvidia-smi; _last_gs on off-ticks) + Prometheus CPU/rest (15 s) x PSU factor:
    # the estimate (rig:power_system_wall_est_watts), kept because it never lags the GPU numbers shown
    # next to it. The card's headline number is wall_w = rig:power_system_wall_watts = measured plug W.
    en = _energy
    if en.get("psu_k") and t - en.get("t", 0) < 60 and point.get("gpu_power_by"):
        point["sys_power"] = round((sum(point["gpu_power_by"]) + (en["cpu_w"] or 0) + (en["other_w"] or 0)) * en["psu_k"], 0)

    svc = None
    booting = any(_boot.get(e) for e in ENG)
    if tick % 5 == 0 or booting or not _ui:
        for eid, (_url, unit) in LANES.items():
            _ui[eid] = unit_info(unit) or _ui.get(eid, {})
        svc = {eid: _ui[eid].get("active") or service_state(LANES[eid][1]) for eid in LANES}
    up_now = {e: raws[e] is not None for e in ENG}
    if tick % 10 == 0 or up_now != _up_prev or not _meta:
        for eid in ENG:
            prev_m = _meta.get(eid) or {}
            m = lane_meta(eid, raws[eid], _ui.get(eid, {}), g or state.get("gpus") or [])
            if raws[eid] is None and prev_m:        # engine down: keep what it last ran, the preset may be mid-edit
                m = {**m, **{k: prev_m[k] for k in ("model", "version", "ctx", "ctx_label", "kv", "slots", "rope",
                                                      "images", "arena_mib") if prev_m.get(k) is not None}}
            _meta[eid] = m
        _names.update(lane_names(_meta))
        _up_prev.clear()
        _up_prev.update(up_now)
    if tick % 2 == 0 or booting:
        for eid in ENG:
            try:
                _boot[eid] = boot_status(eid, raws[eid], _ui.get(eid, {}), _meta.get(eid, {}), _names)
            except Exception as e:
                print("boot status failed:", eid, repr(e), flush=True)
                _boot[eid] = None

    rows = []
    fresh = set()
    for eid in ENG:
        if not raws[eid]:
            continue
        for r in strata_rows(raws[eid], eid):
            rows.append(r)
            if r["ts"] > _req_ts[eid]:
                _req_ts[eid] = r["ts"]
                fresh.add((eid, r.get("id")))
    try:
        hook_events()
        serve_events({eid: _ui.get(eid, {}).get("preset") for eid in ENG})
    except Exception as e:
        print("attribution tail failed:", repr(e), flush=True)
    enrich_rows(rows, fresh)
    point["disk"] = {eid: dict(_dsum[eid]) for eid in ENG}
    for r in rows:
        if (r["eng"], r.get("id")) in fresh:
            db_insert_req(r["ts"], r["eng"], r)

    with lock:
        history.append(point)
        state["eng"] = {eid: {"live": raws[eid] or state["eng"][eid].get("live"),
                              "stats": stats[eid] or state["eng"][eid]["stats"],
                              "meta": _meta.get(eid), "boot": _boot.get(eid)} for eid in ENG}
        state["host"] = h
        state["hw"] = hw
        state["energy"] = _energy
        state["updated"] = t
        if g:
            state["gpus"] = g
        if rows or not state["requests"]:
            # keep the last known rows of a lane that missed one scrape
            seen = {r["eng"] for r in rows}
            keep = [r for r in state["requests"] if r["eng"] not in seen and raws.get(r["eng"]) is None]
            state["requests"] = sorted(rows + keep, key=lambda r: r["ts"] or 0)  # chronological; page sorts newest-first
        if svc:
            state["services"] = svc
    db_insert("hist1", t, point)
    if tick % ROLL_S == 0:
        rollup_and_prune()
    return {}, t


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
                 "sys_power": avg("sys_power"), "vram_gib": avg("vram_gib")}
            for eid in ENG:
                for k in ("_tps", "_prefill_tps", "_ctx_pct"):
                    r[eid + k] = avg(eid + k)
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
            con.execute("DELETE FROM reqs WHERE t < ?", (now - KEEPREQ,))
    except sqlite3.Error as e:
        print("rollup failed:", repr(e), flush=True)


def sampler():
    global _req_ts
    _req_ts = _req_seed()
    prev, prev_t, tick = {}, None, 0
    while True:
        t = time.time()
        try:
            prev, prev_t = sample(prev, prev_t, tick)
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
    ckeys = [CKEY[e] for e in ENG]
    prev_c = {e: None for e in ckeys}
    acc = {e: {"req": 0.0, "prompt": 0.0, "reused": 0.0, "gen": 0.0, "pre_s": 0.0, "dec_s": 0.0} for e in ckeys}
    dacc = {e: {"saves": 0.0, "bytes": 0.0, "saved_tok": 0.0, "restores": 0.0, "rest_tok": 0.0,
                "rest_ms": 0.0, "fails": 0.0} for e in ENG}
    dprev = {e: None for e in ENG}
    for (j,) in rows:
        p = json.loads(j)
        cs = p.get("counters") or {}
        for e in ckeys:
            c1 = cs.get(e)
            if not c1:
                continue
            c0 = prev_c[e]
            if c0:
                for k in acc[e]:
                    d = c1.get(k, 0) - c0.get(k, 0)
                    if d > 0:  # engine restart resets the counters -> skip this delta
                        acc[e][k] += d
            prev_c[e] = c1
        ds = p.get("disk") or {}
        for e in ENG:
            c1 = ds.get(e)
            if not c1:
                continue
            c0 = dprev[e]
            if c0:
                for k in dacc[e]:
                    d = c1.get(k, 0) - c0.get(k, 0)
                    if d > 0:  # monitor restart resets these too: skip the delta
                        dacc[e][k] += d
            dprev[e] = c1
    for eng in ENG:
        s = acc[CKEY[eng]]
        prompt, reused = s["prompt"], s.get("reused", 0)
        dk = dacc[eng]
        out["engines"][eng] = {
            "requests": round(s["req"]), "prompt": round(prompt), "reused": round(reused),
            "computed": round(prompt - reused), "output": round(s["gen"]),
            "prefill_tps": round((prompt - reused) / s["pre_s"]) if s["pre_s"] > 1 else None,
            "decode_tps": round(s["gen"] / s["dec_s"], 1) if s["dec_s"] > 1 else None,
            "disk": {"saves": round(dk["saves"]), "restores": round(dk["restores"]),
                     "fails": round(dk["fails"]), "bytes": round(dk["bytes"]),
                     "saved_tokens": round(dk["saved_tok"]), "restored_tokens": round(dk["rest_tok"]),
                     "avg_restore_ms": round(dk["rest_ms"] / dk["restores"], 1) if dk["restores"] else None,
                     "hit_pct": round(100 * dk["restores"] / s["req"], 1) if s["req"] else None}}
    rq = {e: [] for e in ENG}
    for eng, j in reqs:
        if eng not in rq:   # rows of retired engines stay in the db but are not reported
            continue
        try:
            rq[eng].append(json.loads(j))
        except ValueError:
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


def api_requests(qs):
    """Persisted request rows, newest first: ?range=..|from/to&eng=main|l2|l3&q=<session/conv/status>&limit&offset.
    q filters in python over the JSON fields (session id, conv key, status), so offset skips filtered rows."""
    t0, t1 = window_range(qs)
    eng = (qs.get("eng") or [""])[0]
    q = ((qs.get("q") or [""])[0]).strip().lower()
    try:
        limit = min(1000, max(1, int((qs.get("limit") or ["200"])[0])))
    except ValueError:
        limit = 200
    try:
        offset = max(0, int((qs.get("offset") or ["0"])[0]))
    except ValueError:
        offset = 0
    where, args = "t BETWEEN ? AND ?", [t0, t1]
    if eng in ENG:
        where += " AND eng = ?"
        args.append(eng)
    with DB_LOCK, db() as con:
        if q:
            # text filter spans JSON fields: stream the window newest-first, filter in python
            fetched, total = [], 0
            for e, j in con.execute(f"SELECT eng,j FROM reqs WHERE {where} ORDER BY t DESC", args):
                try:
                    r = json.loads(j)
                except ValueError:
                    continue
                blob = " ".join(str(r.get(k) or "") for k in ("sess", "conv", "status", "id")).lower()
                if q not in blob:
                    continue
                if total >= offset and len(fetched) < limit:
                    fetched.append({**r, "eng": e})
                total += 1
        else:
            total = con.execute(f"SELECT COUNT(*) FROM reqs WHERE {where}", args).fetchone()[0]
            fetched = [{**json.loads(j), "eng": e} for e, j in con.execute(
                f"SELECT eng,j FROM reqs WHERE {where} ORDER BY t DESC LIMIT ? OFFSET ?",
                args + [limit, offset]).fetchall()]
    return {"from": t0, "to": t1, "total": total, "rows": fetched}


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
                m["eng"] = {eid: {"stats": e["stats"], "meta": e.get("meta"), "boot": e.get("boot"),
                                  **({"live": e["live"]} if e.get("live") else {})}
                            for eid, e in state["eng"].items()}
                for eid, e in m["eng"].items():  # main live: the page needs live/engine/cache/totals, not raw metric dumps
                    if "live" in e and isinstance(e["live"], dict):
                        e["live"] = {k: v for k, v in e["live"].items() if k not in ("requests", "history")}
                m["eng_names"] = {eid: _names.get(eid) or ENG[eid] for eid in ENG}
                m["history"] = list(history)
                m["requests"] = state["requests"][-200:]
            self.send(json.dumps(m).encode(), "application/json")
        elif u.path == "/api/history":
            self.send(json.dumps(api_history(qs)).encode(), "application/json")
        elif u.path == "/api/summary":
            self.send(json.dumps(api_summary(qs)).encode(), "application/json")
        elif u.path == "/api/hw":
            self.send(json.dumps(api_hw(qs)).encode(), "application/json")
        elif u.path == "/api/requests":
            self.send(json.dumps(api_requests(qs)).encode(), "application/json")
        elif u.path in ("/", "/index.html"):
            self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif u.path in ("/hw", "/hw.html"):
            self.send(Path(__file__).with_name("hw.html").read_bytes(), "text/html; charset=utf-8")
        elif u.path in ("/history", "/history.html"):
            self.send(Path(__file__).with_name("history.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_error(404)


_energy = {}  # last Prometheus power/cost snapshot (estimate; rules in rig/config/prometheus-rules/power.yml)


def prom_scalar(expr):
    try:
        with urlopen(PROM + "/api/v1/query?" + urlencode({"query": expr}), timeout=5) as r:
            res = json.load(r)["data"]["result"]
        return float(res[0]["value"][1]) if res else None
    except Exception:
        return None


def energy_poller():
    """Every 15 s: system power (wall, estimated), tariff, kWh/RSD totals for today / month / 24h / 30d."""
    global _energy
    while True:
        now = time.time()
        lt = time.localtime(now)
        midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        month0 = time.mktime((lt.tm_year, lt.tm_mon, 1, 0, 0, 0, 0, 0, -1))
        e = {k: prom_scalar(q) for k, q in (
            ("wall_w", "rig:power_system_wall_watts"), ("gpu_w", "rig:power_gpu_watts"),
            ("cpu_w", "rig:power_cpu_watts"), ("other_w", "rig:power_other_watts"),
            ("price", "rig:electricity_price_rsd_per_kwh"), ("rsd_h", "rig:electricity_cost_rsd_per_hour"),
            ("month_proj_rsd", "avg_over_time(rig:electricity_cost_rsd_per_hour[7d]) * 730"),
            ("month_proj_kwh", "avg_over_time(rig:power_system_wall_watts[7d]) * 730 / 1000"))}
        for name, secs in (("today", now - midnight), ("month", now - month0), ("d1", 86400), ("d30", 30 * 86400)):
            w = f"[{max(60, int(secs))}s]"
            e[name + "_kwh"] = prom_scalar(f"sum_over_time(rig:power_system_wall_watts{w}) * 15 / 3600 / 1000")
            e[name + "_rsd"] = prom_scalar(f"sum_over_time(rig:electricity_cost_rsd_per_hour{w}) * 15 / 3600")
        e["t"] = now
        e["ok"] = e["wall_w"] is not None
        dc = sum(e[k] or 0 for k in ("gpu_w", "cpu_w", "other_w"))
        e["psu_k"] = e["wall_w"] / dc if e["ok"] and dc else None  # wall/dc factor (1/PSU efficiency) from the rules
        _energy = e
        time.sleep(15)


db_init()
db_backfill()
threading.Thread(target=energy_poller, daemon=True).start()
threading.Thread(target=sampler, daemon=True).start()
ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
