#!/usr/bin/env python3
"""Junction (hotspot) fan booster for the RTX 3090 blowers.

The VBIOS curve (Vix payload M4, docs/07) follows core temp only; core sits at the
83 C boost target, so fans stay ~84 % while junction reaches 100-104 C.
Here: junction >= HI -> force 100 % via NVML; junction < LO -> hand back to the
VBIOS curve (nvmlDeviceSetDefaultFanSpeed_v2). Junction comes from the gputemps
exporter textfile (NVML does not expose hotspot on GeForce). Stale/missing data
leaves the current state as is. SIGTERM / --reset restores the VBIOS curve.
"""
import os, re, signal, sys, time
import pynvml as n

HI, LO = 95, 90          # C, hysteresis
ON_S, HOLD_S = 20, 180   # junction must stay >= HI this long to boost; boost lasts at least HOLD_S
PROM = "/var/lib/prometheus/node-exporter/gputemps.prom"
STALE = 30               # s; exporter refreshes every 5 s
MATCH = "3090"
RX = re.compile(r'sensor="junction",uuid="([^"]+)".*\}\s+([\d.]+)')


def decide(st, junction, now):
    """st = {'on': bool, 'since': t of last state change, 'hot': t junction first >= HI or None}.
    Junction swings 80<->97 C within seconds under bursty decode, so boost only after
    ON_S of sustained heat and keep it HOLD_S minimum. None junction keeps the state."""
    if junction is None:
        return st["on"]
    if junction < HI:
        st["hot"] = None
    elif st["hot"] is None:
        st["hot"] = now
    if not st["on"]:
        return st["hot"] is not None and now - st["hot"] >= ON_S
    return junction >= LO or now - st["since"] < HOLD_S


def junctions():
    try:
        if time.time() - os.path.getmtime(PROM) > STALE:
            return {}
        return {u: float(t) for u, t in RX.findall(open(PROM).read())}
    except OSError:
        return {}


def gpus():
    for i in range(n.nvmlDeviceGetCount()):
        h = n.nvmlDeviceGetHandleByIndex(i)
        if MATCH in n.nvmlDeviceGetName(h):
            yield n.nvmlDeviceGetUUID(h).removeprefix("GPU-"), h


def apply(h, boost):
    for f in range(n.nvmlDeviceGetNumFans(h)):
        try:
            if boost:
                n.nvmlDeviceSetFanSpeed_v2(h, f, 100)
            else:
                n.nvmlDeviceSetDefaultFanSpeed_v2(h, f)
        except n.NVMLError as e:
            print(f"fan {f}: {e}", file=sys.stderr, flush=True)


def reset(*_):
    for _u, h in gpus():
        apply(h, False)
    sys.exit(0)


def selftest():
    def run(seq, st=None):  # seq of (t, junction) -> list of on/off
        st = st or {"on": False, "since": 0, "hot": None}
        out = []
        for t, j in seq:
            new = decide(st, j, t)
            if new != st["on"]:
                st["on"], st["since"] = new, t
            out.append(new)
        return out
    # 15 s spike -> no boost
    assert not any(run([(0, 80), (5, 97), (10, 97), (15, 96), (20, 82)]))
    # sustained heat -> boost at 20 s
    assert run([(0, 96), (10, 96), (20, 96)]) == [False, False, True]
    # boosted: drop below LO inside HOLD keeps it on, after HOLD turns off
    r = run([(0, 96), (20, 96), (60, 85), (199, 85), (201, 85)])
    assert r == [False, True, True, True, False], r
    # stays on while junction >= LO even after HOLD; None keeps state
    assert run([(0, 96), (20, 96), (300, 91), (305, None)])[-2:] == [True, True]
    print("ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    n.nvmlInit()
    if "--reset" in sys.argv:
        reset()
    signal.signal(signal.SIGTERM, reset)
    signal.signal(signal.SIGINT, reset)
    state = {}
    while True:
        j, now = junctions(), time.time()
        for uuid, h in gpus():
            st = state.setdefault(uuid, {"on": False, "since": now, "hot": None})
            new = decide(st, j.get(uuid), now)
            if new != st["on"]:
                print(f"{uuid[:8]} junction {j.get(uuid)} -> {'100%' if new else 'VBIOS curve'}", flush=True)
                apply(h, new)
                st["on"], st["since"] = new, now
            elif new:
                apply(h, True)   # re-assert: driver may drop a manual setting
        time.sleep(5)
