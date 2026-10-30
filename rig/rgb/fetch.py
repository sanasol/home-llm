#!/usr/bin/env python3
"""Fetch a real Prometheus window and save it as replay frames for rgbshow.py."""
import json, sys, time, urllib.request, urllib.parse

P = "http://localhost:9090/api/v1/query_range?"
UUID = {"5565fcfc-574c-5daa-bd59-e29329f6e466": "0",
        "2e674bbf-af1e-4ad3-17f6-abc8e47b4f36": "1",
        "9c7268f4-61e4-a63f-7a6f-e33d57743219": "2"}


def qr(expr, start, end, step):
    u = P + urllib.parse.urlencode({"query": expr, "start": start, "end": end, "step": step})
    return json.load(urllib.request.urlopen(u))["data"]["result"]


def series(expr, start, end, step, key):
    out = {}
    for s in qr(expr, start, end, step):
        k = key(s["metric"])
        out[k] = {int(float(t)): float(v) for t, v in s["values"]}
    return out


def main():
    start = time.mktime(time.strptime(sys.argv[1], "%Y-%m-%d %H:%M"))
    end = time.mktime(time.strptime(sys.argv[2], "%Y-%m-%d %H:%M"))
    step = int(sys.argv[3])
    out = sys.argv[4]
    gi = lambda m: m.get("gpu_index", "?")
    uu = lambda m: UUID.get(m.get("uuid"), "?")
    one = lambda m: "x"
    S = {
        "tps": series("rate(rig_llm_generation_tokens_total[15s])", start, end, step, gi),
        "run": series("rig_llm_requests_running", start, end, step, gi),
        "req": series("rig_llm_requests_total", start, end, step, gi),
        "up": series("rig_llm_up", start, end, step, gi),
        "util": series("nvidia_smi_utilization_gpu_ratio", start, end, step, uu),
        "temp": series("nvidia_smi_temperature_gpu", start, end, step, uu),
        "cpu": series('max(node_hwmon_temp_celsius{chip=~".*18_3.*"})', start, end, step, one),
        "mem": series("1-node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes", start, end, step, one),
        "power": series("rig:power_gpu_watts", start, end, step, one),
    }
    frames, prev_req = [], {}
    t = int(start)
    while t <= end:
        f = {"t": t}
        for name, d in S.items():
            f[name] = {k: v[t] for k, v in d.items() if t in v}
        newreq = {}
        for k, v in f["req"].items():
            newreq[k] = max(0.0, v - prev_req.get(k, v))
            prev_req[k] = v
        f["newreq"] = newreq
        del f["req"]
        frames.append(f)
        t += step
    json.dump({"step": step, "frames": frames}, open(out, "w"))
    peak = max(sum(f["tps"].values()) for f in frames)
    print(f"{len(frames)} frames, peak total {peak:.0f} t/s, "
          f"max temp {max(max(f['temp'].values() or [0]) for f in frames):.0f}C, "
          f"cpu max {max(f['cpu'].get('x', 0) for f in frames):.0f}C, "
          f"engine-down frames {sum(1 for f in frames if 0.0 in f['up'].values())}")


if __name__ == "__main__":
    main()
