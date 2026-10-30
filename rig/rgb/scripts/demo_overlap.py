#!/usr/bin/env python3
"""RGB overlap demo: synthetic frames through rgbshow mode-2 replay — no real
traffic. Sequence on the 5090 strip (30 LEDs) and RAM sticks (12+12):
  0-5s   decode: green bar rises
  5-11s  OVERLAP: green decode + orange prefill stacked together
  11-17s prefill only: orange (what the engine actually does today)
  17-21s decode again: green
  21-24s idle: off
usage on the rig:
  python3 ~/rgb/demo_overlap.py            # writes /tmp/rgb_demo.json
  python3 ~/rgb/rgbshow.py 2 /tmp/rgb_demo.json 1
  sudo systemctl restart rig-rgb                  # back to live
"""
import json

STEP = 0.5
T = 24.0
frames = []
t = 0.0
prev = {}


def seg(t0, t1, f):
    """emit frames for [t0,t1) with value fn f(t) -> (dec5090, pre5090, decL, preL)"""
    global t
    while t < t1 - 1e-9:
        k = (t - t0) / (t1 - t0)
        d0, p0, dL, pL = f(k)
        newreq = {g: (6.0 if t0 == prev.get("t0", -1) else 0.0) for g in "012"}
        frames.append({
            "t": int(t * 10),  # only used as a label in replay status
            "tps": {"0": d0, "1": dL, "2": dL},
            "pre": {"0": p0, "1": pL, "2": pL},
            "run": {"0": 1 if (d0 or p0) else 0, "1": 1 if (dL or pL) else 0, "2": 1 if (dL or pL) else 0},
            "phase": {"0": 3 if (d0 and p0) else (2 if d0 else (1 if p0 else 0)),
                      "1": 3 if (dL and pL) else (2 if dL else (1 if pL else 0)),
                      "2": 3 if (dL and pL) else (2 if dL else (1 if pL else 0))},
            "up": {"0": 1, "1": 1, "2": 1},
            "util": {"0": 0.9, "1": 0.7, "2": 0.7},
            "temp": {"0": 68, "1": 61, "2": 59},
            "cpu": {"x": 52}, "mem": {"x": 0.45}, "newreq": newreq, "alerts": {},
        })
        prev["t0"] = t0
        t += STEP


seg(0, 5, lambda k: (200 * min(1, k * 1.5), 0, 110 * min(1, k * 1.5), 0))
seg(5, 11, lambda k: (170, 7000 * min(1, k * 1.4), 95, 6500 * min(1, k * 1.4)))
seg(11, 17, lambda k: (0, 7000 * (1.0 if k < 0.7 else max(0, 1 - (k - 0.7) / 0.3)),
                       0, 6500 * (1.0 if k < 0.7 else max(0, 1 - (k - 0.7) / 0.3))))
seg(17, 21, lambda k: (150 * (1 - k) + 0, 0, 100 * (1 - k), 0))
seg(21, 24, lambda k: (0, 0, 0, 0))

json.dump({"step": STEP, "frames": frames}, open("/tmp/rgb_demo.json", "w"))
print(f"wrote /tmp/rgb_demo.json: {len(frames)} frames, {T:.0f}s loop")
