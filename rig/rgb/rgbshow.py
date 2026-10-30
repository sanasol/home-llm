#!/usr/bin/env python3
"""Rig RGB show: replay real Prometheus frames (or live) through 5 light modes.

usage: rgbshow.py MODE REPLAY.json SPEEDUP
modes: 1 per-device load, 2 LLM equalizer, 3 speed comet, 4 thermal, 5 alerts
"""
import json, math, sys, time
from openrgb import OpenRGBClient
from openrgb.utils import RGBColor, DeviceType

MODE = int(sys.argv[1])
LIVE = sys.argv[2] == "live"
if LIVE:
    FR, STEP, SPEED = [], 1, 1.0
else:
    DATA = json.load(open(sys.argv[2]))
    SPEED = float(sys.argv[3])
    FR, STEP = DATA["frames"], DATA["step"]
STATUS = "/tmp/rgbshow.status"

TEMP_PAL = [(40, (0, 80, 255)), (60, (0, 255, 120)), (72, (255, 200, 0)),
            (80, (255, 90, 0)), (88, (255, 0, 0))]
LOAD_PAL = [(0.0, (0, 40, 255)), (0.5, (160, 0, 255)), (1.0, (255, 0, 40))]
RUN_COL = [(0, 200, 255), (0, 255, 80), (255, 220, 0), (255, 0, 200)]
DEC_FULL = {"0": 250.0, "1": 130.0, "2": 130.0}      # decode t/s full-scale per lane
PRE_FULL = {"0": 7000.0, "1": 6500.0, "2": 6500.0}   # prefill t/s full-scale per lane
DEC_COL, PRE_COL = (0, 255, 80), (255, 140, 0)       # green decode / orange prefill
OFF = (0, 0, 0)


def grad(pal, x):
    if x <= pal[0][0]:
        return pal[0][1]
    for (a, ca), (b, cb) in zip(pal, pal[1:]):
        if x <= b:
            k = (x - a) / (b - a)
            return tuple(int(ca[i] + (cb[i] - ca[i]) * k) for i in range(3))
    return pal[-1][1]


def scale(c, k):
    k = max(0.0, min(1.0, k))
    return tuple(int(v * k) for v in c)


def hsv(h, s=1.0, v=1.0):
    h = (h % 1.0) * 6
    i, f = int(h), h - int(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return (int(r * 255), int(g * 255), int(b * 255))


def bar(n_leds, fill, on, off=(0, 0, 0)):
    fill = max(0.0, min(1.0, fill)) * n_leds
    out = []
    for i in range(n_leds):
        k = max(0.0, min(1.0, fill - i))
        out.append(tuple(int(off[j] + (on[j] - off[j]) * k) for j in range(3)))
    return out


class Rig:
    def __init__(self):
        for attempt in range(60):  # server may still be detecting devices after boot
            try:
                self.c = OpenRGBClient("127.0.0.1", 6742, "rigshow")
                if len(self.c.devices) >= 3:
                    break
                self.c.disconnect()
            except Exception:
                pass
            time.sleep(2)
        else:
            raise SystemExit("OpenRGB server not ready")
        self.ram = self.c.get_devices_by_type(DeviceType.DRAM)[0]
        self.gpu = self.c.get_devices_by_type(DeviceType.GPU)[0]
        self.mb = self.c.get_devices_by_type(DeviceType.MOTHERBOARD)[0]
        for d in (self.ram, self.gpu, self.mb):
            d.set_mode("direct")
        self.zl = [len(z.leds) for z in self.mb.zones]  # zone0 RGB, 1 ARGB1, 2 pump, 3 fans
        self.last = {}

    def push(self, name, dev, cols):
        if self.last.get(name) == cols:
            return
        self.last[name] = cols
        dev.set_colors([RGBColor(*c) for c in cols], fast=True)

    def show(self, tick, gpu30, ram_a12, ram_b12, pump, fans):
        if tick % 2 == 0:
            self.push("gpu", self.gpu, gpu30)
        if tick % 4 == 0:
            self.push("ram", self.ram, ram_a12 + ram_b12)
        mb = [OFF] * self.zl[0] + [OFF] * self.zl[1]
        mb += pump if isinstance(pump, list) else [pump] * self.zl[2]
        mb += fans if isinstance(fans, list) else [fans] * self.zl[3]
        self.push("mb", self.mb, mb)


class Live:
    """Real-time metrics for RGB: poll the Strata engines DIRECTLY at 4 Hz
    (phase, rates, request bursts) + NVML/Grafana/system at 1 Hz. The Prometheus
    exporter gauges are 15 s active-means — fine for graphs, unusable for lights.
    Strata live-block semantics (probed 30.10): tok_s = live decode rate only while
    `generating`; prefill progress is only reported per 32K chunk, so prefill rate
    comes from totals.prompt_tokens deltas (EMA-smoothed) and while `reading` before
    the first chunk lands the bar gets a floor value. Reads keys from launch
    presets; never prints them."""
    LANES = {"0": (8001, "main.json"), "1": (8003, "lane2.json"), "2": (8004, "lane3.json")}
    POLL = 0.25
    PRE_MIN = {"0": 1200.0, "1": 1000.0, "2": 1000.0}  # orange floor while reading, t/s

    def __init__(self):
        import os, pynvml, threading, urllib.request
        self.nv, self.url = pynvml, urllib.request
        pynvml.nvmlInit()
        self.h = {str(i): pynvml.nvmlDeviceGetHandleByIndex(i)
                  for i in range(pynvml.nvmlDeviceGetCount())}
        d = os.path.expanduser("~/.config/strata/launch-presets")
        self.eng, self.cur = {}, {}
        for g, (port, preset) in self.LANES.items():
            try:
                key = json.load(open(f"{d}/{preset}"))["api_key"]
                self.eng[g] = {"url": f"http://127.0.0.1:{port}", "key": key,
                               "ptok": 0.0, "otok": 0.0, "req": 0.0, "gen": 0.0,
                               "ema": 0.0, "t": 0.0, "init": False, "fails": 0}
            except Exception:
                self.eng[g] = None
            self.cur[g] = {"tps": 0.0, "pre": 0.0, "run": 0.0, "phase": 0.0,
                           "up": None, "newreq": 0.0}
        self.slow = self.slow_poll()
        self.frames = [self.frame(), self.frame()]
        self.ts, self.seq = time.time(), 0
        threading.Thread(target=self.fast_loop, daemon=True).start()
        threading.Thread(target=self.slow_loop, daemon=True).start()

    def fast_tick(self):
        for g, e in self.eng.items():
            if e is None:
                continue
            try:
                rq = self.url.Request(e["url"] + "/metrics",
                                      headers={"Authorization": f"Bearer {e['key']}"})
                d = json.loads(self.url.urlopen(rq, timeout=1.0).read())
                tot, live = d.get("totals", {}), d.get("live", {})
                now = time.time()
                dt = max(0.05, now - e["t"]) if e["t"] else 1.0
                pt = float(tot.get("prompt_tokens", 0))
                ot = float(tot.get("output_tokens", 0))
                reqs = float(tot.get("requests", 0))
                if not e["init"]:  # first read: baseline counters, no fake deltas
                    e.update(ptok=pt, otok=ot, req=reqs, gen=0.0, init=True, t=now)
                    continue
                dpt, dot = max(0.0, pt - e["ptok"]), max(0.0, ot - e["otok"])
                gen = live.get("generated")
                dotg = max(0.0, gen - e["gen"]) if gen is not None else 0.0
                if gen is not None:
                    e["gen"] = gen
                e["ptok"], e["otok"] = pt, ot
                st = live.get("state")
                # totals.output_tokens only bumps at request END; live.generated
                # bumps per step -> use it (or tok_s) for live decode rate/bit
                ph = (1 if st == "reading" else 0) | (2 if (st == "generating" or dotg > 0 or dot > 0) else 0)
                ts = live.get("tok_s")
                dec = ts if (ts and st == "generating") else (max(dotg, dot) / dt if (dotg or dot) else 0.0)
                if ph & 1:
                    e["ema"] = 0.7 * e["ema"] + 0.3 * (dpt / dt)
                    pre = max(e["ema"], self.PRE_MIN[g])
                else:
                    e["ema"], pre = 0.0, 0.0
                self.cur[g] = {"tps": dec, "pre": pre,
                               "run": 0.0 if st in (None, "idle") else 1.0,
                               "phase": float(ph), "up": 1.0,
                               "newreq": max(0.0, reqs - e["req"])}
                e["req"], e["t"], e["fails"] = reqs, now, 0
            except Exception:
                e["fails"] = e.get("fails", 0) + 1
                if e["fails"] >= 3:  # sustained failure = engine down (red pulse);
                    self.cur[g] = {"tps": 0.0, "pre": 0.0, "run": 0.0, "phase": 0.0,
                                   "up": 0.0, "newreq": 0.0}
                # 1-2 misses: keep last frame (slow /metrics under load, not downtime)

    def fast_loop(self):
        while True:
            time.sleep(max(0.02, self.POLL - (time.time() - self.ts)))
            self.fast_tick()
            self.frames = [self.frames[1], self.frame()]
            self.ts, self.seq = time.time(), self.seq + 1

    def slow_loop(self):
        while True:
            time.sleep(1.0)
            self.slow = self.slow_poll()

    def slow_poll(self):
        s = {"util": {}, "temp": {}, "cpu": {}, "mem": {}, "alerts": {}}
        for g, h in self.h.items():
            try:
                s["util"][g] = self.nv.nvmlDeviceGetUtilizationRates(h).gpu / 100
                s["temp"][g] = float(self.nv.nvmlDeviceGetTemperature(h, 0))
            except Exception:
                pass
        try:  # Grafana alerting state (unauthenticated /metrics on loopback)
            txt = self.url.urlopen("http://127.0.0.1:3000/metrics", timeout=0.8).read().decode()
            i = txt.find('grafana_alerting_alerts{state="alerting"} ')
            if i >= 0:
                s["alerts"]["x"] = float(txt[i + 44:].split(None, 1)[0])
        except Exception:
            pass
        try:
            for hw in __import__("glob").glob("/sys/class/hwmon/hwmon*"):
                if open(hw + "/name").read().strip() == "k10temp":
                    s["cpu"]["x"] = int(open(hw + "/temp1_input").read()) / 1000
            mi = dict(l.split(":", 1) for l in open("/proc/meminfo"))
            tot, av = int(mi["MemTotal"].split()[0]), int(mi["MemAvailable"].split()[0])
            s["mem"]["x"] = 1 - av / tot
        except Exception:
            pass
        return s

    def frame(self):
        f = {k: {} for k in ("tps", "pre", "run", "phase", "up", "util", "temp",
                             "cpu", "mem", "newreq")}
        for g, c in self.cur.items():
            f["tps"][g], f["pre"][g], f["run"][g] = c["tps"], c["pre"], c["run"]
            f["phase"][g], f["up"][g], f["newreq"][g] = c["phase"], c["up"], c["newreq"]
        for k in ("util", "temp", "cpu", "mem"):
            f[k] = dict(self.slow[k])
        f["alerts"] = dict(self.slow["alerts"])
        f["t"] = int(time.time())
        return f


LIVE_SRC = None


def frame_at(rt):
    if LIVE:
        a, b = LIVE_SRC.frames
        k = min(1.0, (time.time() - LIVE_SRC.ts) / Live.POLL)
        i = LIVE_SRC.seq
    else:
        i = min(int(rt // STEP), len(FR) - 2)
        a, b, k = FR[i], FR[i + 1], (rt - i * STEP) / STEP
    return _interp(a, b, k, i)


def _interp(a, b, k, i):
    """Interpolate metrics between frames a and b; events come from the newer frame."""

    def m(name, key, default=0.0):
        va, vb = a[name].get(key), b[name].get(key)
        if va is None and vb is None:
            return default
        va = vb if va is None else va
        vb = va if vb is None else vb
        return va + (vb - va) * k
    return (b if LIVE else a), i, m


def main():
    global LIVE_SRC
    rig = Rig()
    if LIVE:
        LIVE_SRC = Live()
    t0 = time.time()
    span = max(1, (len(FR) - 1) * STEP)
    pos, flash, last_i, hue = 0.0, 0.0, -1, 0.0
    tick, prev = 0, time.time()
    while True:
        now = time.time()
        dt, prev = now - prev, now
        rt = ((now - t0) * SPEED) % span
        a, i, m = frame_at(rt)
        if i != last_i:  # passed real frames -> collect request bursts
            newreq = sum(a["newreq"].values())
            flash = min(1.0, flash + newreq / 6.0)
            last_i = i
        flash *= math.exp(-dt * 3)
        tps = {g: m("tps", g) for g in "012"}
        pre = {g: m("pre", g) for g in "012"}
        run = {g: m("run", g) for g in "012"}
        temp = {g: m("temp", g, 40) for g in "012"}
        util0 = m("util", "0")
        up = {g: a["up"].get(g) for g in "012"}  # None = exporter gap
        cpu, mem = m("cpu", "x", 40), m("mem", "x")
        total = sum(tps.values())
        hot = max(list(temp.values()) + [cpu])
        pulse = 0.5 + 0.5 * math.sin(now * 2 * math.pi * 1.5)

        if MODE == 1:
            gpu = [scale(grad(LOAD_PAL, util0), 0.25 + 0.75 * util0)] * 30
            rb = bar(12, mem, grad(LOAD_PAL, mem))
            out = (gpu, rb, rb, grad(TEMP_PAL, cpu), grad(TEMP_PAL, max(temp.values())))
        elif MODE == 2:
            # bar = activity, color = phase: green decode (full = lane max t/s),
            # orange prefill stacked on top (full = lane max prefill t/s)
            def bar_stack(n, f1, c1, f2, c2):
                gcols = bar(n, f1, c1)
                shift = int(round(max(0.0, min(1.0, f1)) * n))
                ocols = bar(n, f2, c2)
                ocols = [OFF] * shift + ocols[: n - shift]
                return [tuple(max(x, y) for x, y in zip(gc, oc)) for gc, oc in zip(gcols, ocols)]

            def eq(n, g):
                if up[g] == 0.0:
                    return [scale((255, 0, 0), 0.15 + 0.5 * pulse)] * n
                return bar_stack(n, tps[g] / DEC_FULL[g], DEC_COL, pre[g] / PRE_FULL[g], PRE_COL)
            busy = any(run[g] > 0.05 for g in "012")
            base = (0, 40, 50) if busy else OFF
            fl = tuple(int(base[j] + (255 - base[j]) * flash) for j in range(3))
            pump = fl
            if a.get("alerts", {}).get("x", 0) > 0:  # firing Grafana alert -> pump blinks red
                pump = (255, 0, 0) if int(now * 2) % 2 == 0 else OFF
            out = (eq(30, "0"), eq(12, "2"), eq(12, "1"), pump, fl)
        elif MODE == 3:
            if total < 5:
                br = 0.08 + 0.25 * (0.5 + 0.5 * math.sin(now * 1.2))
                c = scale((0, 255, 200), br)
                out = ([c] * 30, [c] * 12, [c] * 12, c, c)
            else:
                speed = 3 + total / 1000 * 45  # LEDs per second
                pos = (pos + speed * dt) % 30
                hue = (hue + dt * (0.02 + total / 1000 * 0.4)) % 1
                gpu = []
                for j in range(30):
                    d = (pos - j) % 30
                    gpu.append(hsv(hue, 1, max(0.03, math.exp(-d / 3))))
                rc = hsv(hue + 0.33, 1, 0.2 + 0.8 * min(1, total / 900))
                pump = [hsv(hue + j / 24, 1, 1) for j in range(rig.zl[2])]
                fans = [hsv(hue + 0.5 + j / 24, 1, 1) for j in range(rig.zl[3])]
                out = (gpu, [rc] * 12, [rc] * 12, pump, fans)
        elif MODE == 4:
            c = grad(TEMP_PAL, hot)
            if hot >= 85:
                c = scale(c, 0.3 + 0.7 * pulse)
            out = ([c] * 30, [c] * 12, [c] * 12, c, c)
        else:  # MODE 5: dark unless something is wrong
            purple = scale((180, 0, 255), 0.2 + 0.8 * pulse)
            gpu = [purple] * 30 if up["0"] == 0.0 else [OFF] * 30
            ra = [purple] * 12 if up["2"] == 0.0 else [OFF] * 12
            rb = [purple] * 12 if up["1"] == 0.0 else [OFF] * 12
            warm = scale((255, 70, 0), (hot - 80) / 5) if hot > 80 else OFF
            out = (gpu, ra, rb, OFF, warm)

        rig.show(tick, *out)
        if tick % 10 == 0:
            with open(STATUS, "w") as f:
                f.write(f"mode={MODE} {'LIVE' if LIVE else 'replay'}="
                        f"{time.strftime('%m-%d %H:%M:%S', time.localtime(a['t'] if LIVE else FR[0]['t'] + rt))} "
                        f"tps5090={tps['0']:.0f} tpsL2={tps['1']:.0f} tpsL3={tps['2']:.0f} "
                        f"pre5090={pre['0']:.0f} preL2={pre['1']:.0f} preL3={pre['2']:.0f} "
                        f"run={run['0']:.0f}/{run['1']:.0f}/{run['2']:.0f} hot={hot:.0f}C cpu={cpu:.0f}C "
                        f"up={up['0']}/{up['1']}/{up['2']} flash={flash:.2f} "
                        f"alerts={a.get('alerts', {}).get('x', 0):.0f}\n")
        tick += 1
        time.sleep(0.05)


if __name__ == "__main__":
    main()
