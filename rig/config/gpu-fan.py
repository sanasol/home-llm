#!/usr/bin/env python3
"""Кривая вентилятора для 3090 (blower): заводская держит 41 % даже в простое.
Остальные карты не трогает. При остановке и с --reset возвращает заводскую кривую."""
import signal, sys, time
import pynvml as n

# (°C, %): до 50 °C — минимум карты, к 82 °C — 100 %
CURVE = [(50, 30), (65, 45), (75, 70), (82, 100)]
MATCH = "3090"

def target(t):
    if t <= CURVE[0][0]:
        return CURVE[0][1]
    for (t0, p0), (t1, p1) in zip(CURVE, CURVE[1:]):
        if t <= t1:
            return round(p0 + (p1 - p0) * (t - t0) / (t1 - t0))
    return 100

def gpus():
    for i in range(n.nvmlDeviceGetCount()):
        h = n.nvmlDeviceGetHandleByIndex(i)
        if MATCH in n.nvmlDeviceGetName(h):
            yield h

def fans(h):
    return range(n.nvmlDeviceGetNumFans(h))

def reset(*_):
    for h in gpus():
        for f in fans(h):
            try:
                n.nvmlDeviceSetDefaultFanSpeed_v2(h, f)
            except n.NVMLError:
                pass
    sys.exit(0)

n.nvmlInit()
if "--reset" in sys.argv:
    reset()
signal.signal(signal.SIGTERM, reset)
signal.signal(signal.SIGINT, reset)
last = {}
while True:
    for h in gpus():
        uuid = n.nvmlDeviceGetUUID(h)
        t = n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)
        want = target(t)
        # вниз — только при разнице от 5 %, чтобы турбина не дёргалась туда-сюда
        if want > last.get(uuid, -1) or last[uuid] - want >= 5:
            for f in fans(h):
                try:
                    n.nvmlDeviceSetFanSpeed_v2(h, f, want)
                except n.NVMLError:
                    pass
            last[uuid] = want
    time.sleep(2)
