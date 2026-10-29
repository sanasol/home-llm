#!/usr/bin/env python3
"""Export GPU core/junction(hotspot)/VRAM temps to the node_exporter textfile collector.

nvidia-smi/NVML report 'Memory Current Temp: N/A' on GeForce, so GDDR6X/GDDR7 and
hotspot temps come from gputemps (ThomasBaruzier/gddr6-core-junction-vram-temps),
which reads undocumented MMIO sensors via /dev/mem (needs root; works on the rig
without iomem=relaxed because CONFIG_IO_STRICT_DEVMEM is unset).

Metric: rig_gpu_sensor_temp_celsius{sensor="core|junction|vram", uuid, index, name}
uuid has no "GPU-" prefix, matching nvidia_smi_* labels from nvidia_gpu_exporter.
"""
import json, os, subprocess, sys, time

BIN = os.environ.get("GPUTEMPS_BIN", "/opt/rig/bin/gputemps")
OUT = os.environ.get("GPUTEMPS_OUT", "/var/lib/prometheus/node-exporter/gputemps.prom")
REFRESH_MS = int(os.environ.get("GPUTEMPS_REFRESH_MS", "5000"))


def gpu_ids():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30, check=True)
    ids = {}
    for line in r.stdout.strip().splitlines():
        idx, uuid, name = [x.strip() for x in line.split(",", 2)]
        ids[int(idx)] = (uuid.removeprefix("GPU-"), name)
    return ids


def write(sample, ids):
    lines = ["# HELP rig_gpu_sensor_temp_celsius GPU core/junction/VRAM temperature (gputemps MMIO).",
             "# TYPE rig_gpu_sensor_temp_celsius gauge"]
    for g in sample.get("gpus", []):
        uuid, name = ids.get(g["index"], ("unknown", "unknown"))
        for sensor in ("core", "junction", "vram"):
            v = g.get(sensor)
            if v is None:
                continue
            lines.append(f'rig_gpu_sensor_temp_celsius{{sensor="{sensor}",uuid="{uuid}",'
                         f'index="{g["index"]}",name="{name}"}} {v}')
    lines += ["# HELP rig_gputemps_last_success_timestamp_seconds Last successful gputemps sample.",
              "# TYPE rig_gputemps_last_success_timestamp_seconds gauge",
              f"rig_gputemps_last_success_timestamp_seconds {time.time():.0f}", ""]
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
    os.chmod(tmp, 0o644)
    os.replace(tmp, OUT)


def main():
    ids = gpu_ids()
    p = subprocess.Popen([BIN, "--json", "--refresh-ms", str(REFRESH_MS)],
                         stdout=subprocess.PIPE, text=True, bufsize=1)
    assert p.stdout is not None
    for line in p.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            write(json.loads(line), ids)
        except (ValueError, KeyError, OSError) as e:
            print(f"bad sample: {e}: {line[:200]}", file=sys.stderr, flush=True)
    sys.exit(f"gputemps exited rc={p.wait()}")  # systemd restarts us


if __name__ == "__main__":
    main()
