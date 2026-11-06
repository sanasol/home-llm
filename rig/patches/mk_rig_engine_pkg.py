#!/usr/bin/env python3
"""Build the rig's Strata engine package = a signed vendor package with the `strata` binary swapped for our
self-build (vendor tag + PR #3341 disk conversation cache), as an UNSIGNED sibling package.

usage (ON the rig, as al):
  mk_rig_engine_pkg.py VENDOR_VER RIG_VER BUILD_BINARY
  e.g. mk_rig_engine_pkg.py 0.1.57 0.1.57.1 ~/build/Strata-0211/build/strata

RIG_VER must sort ABOVE the vendor version (the resolver takes the highest usable package):
0.1.57.1 > 0.1.57 (control's _version_key splits on . - +). A later vendor 0.1.58 would win again ->
rebuild the PR package on top of it before restarting the lanes.

Mechanism notes (see mk_pr3341_engine_pkg.py for the 0.1.56 story):
- editing a signed manifest breaks its ed25519 signature and the resolver SILENTLY falls back; unsigned
  packages are accepted unless STRATA_REQUIRE_SIGNED is set -> drop the "signature" field.
- the vendor package's other files (libstrata_kq.so, strata-vision, cuBLAS) stay hardlinked: they match the
  vendor tag our binary was built from.
- `cp -al` + writing a file through the hardlink corrupts the vendor package: unlink BEFORE writing.
- archs[] filters per HOST (5090 + 3090): the build must carry sm_86 AND sm_120
  (cmake -DCMAKE_CUDA_ARCHITECTURES="86;120;120-virtual" -DSTRATA_ENABLE_CUDA=ON).
Rollback: move the RIG_VER folder out of components/engine; lanes fall back to the vendor package on restart
(but then the presets' --conversation-cache-disk-mib / --conversation-cache-dir flags are unknown to it).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

vendor, rig, binary = sys.argv[1], sys.argv[2], os.path.expanduser(sys.argv[3])
BASE = os.path.expanduser("~/.local/share/strata/components/engine")
SUFFIX = "-linux-x86_64-cuda-cuda13"
SRC, DST = f"{BASE}/{vendor}{SUFFIX}", f"{BASE}/{rig}{SUFFIX}"

if not os.path.isdir(SRC):
    sys.exit(f"vendor package missing: {SRC}")
if os.path.exists(DST):
    shutil.rmtree(DST)
subprocess.check_call(["cp", "-al", SRC, DST])
for rel in ("strata", "engine-component.json"):          # break the links we will write
    os.unlink(os.path.join(DST, rel))
shutil.copy2(binary, DST + "/strata")
shutil.copy2(SRC + "/engine-component.json", DST + "/engine-component.json")

h = hashlib.sha256()
with open(DST + "/strata", "rb") as f:
    for chunk in iter(lambda: f.read(1 << 22), b""):
        h.update(chunk)
j = json.load(open(DST + "/engine-component.json"))
j["version"] = rig
j["build_id"] = f"v{vendor}+pr3341-disk-cache"
for e in j["files"]:
    if e["path"] == "strata":
        e["sha256"] = h.hexdigest()
        e["bytes"] = os.path.getsize(DST + "/strata")
j.pop("signature", None)
with open(DST + "/engine-component.json", "w") as f:
    json.dump(j, f, indent=1)
print("built", DST)

ctl = json.load(open(os.path.expanduser("~/.local/share/strata/components/control/active.json")))["folder"]
sys.path.insert(0, os.path.expanduser(f"~/.local/share/strata/components/control/{ctl}"))
from pathlib import Path  # noqa: E402
from strata_control import engine_component as ec  # noqa: E402
rep = ec.status(Path(os.path.expanduser("~/.local/share/strata/components")))
print("control", ctl, "| usable:", rep["usable"], "->", (rep.get("engine") or {}).get("version"))
