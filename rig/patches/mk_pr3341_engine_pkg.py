#!/usr/bin/env python3
"""Build Strata engine package 0.1.56 = v0.2.9 + PR #3341 (disk conversation cache)
for the rig, as an UNSIGNED sibling package so the signed 0.1.55 stays pristine.

Mechanism notes (learned the hard way, 02.11):
- engine_component.resolve() runs verify_signature()+verify() per package. Editing a
  vendor package's manifest (e.g. after swapping its binary) breaks the ed25519
  signature and the resolver SILENTLY falls back to the previous version.
- Unsigned packages are ACCEPTED unless STRATA_REQUIRE_SIGNED is set — so the PR
  package ships with the "signature" field removed.
- scan() parses any `engine/<ver>-<platform>-<backend>` dir; a ".off" suffix does NOT
  hide it (name still parses, backend match is lenient). To remove a package from
  resolution, move it OUT of components/engine entirely.
- The launcher caches the chosen engine path in ~/.local/state/strata/launch/<lane>.json;
  after changing packages, verify the "engine started:" line in <lane>.engine.log.
- archs[] in the manifest filters per HOST, not per lane: nvidia-smi ignores
  CUDA_VISIBLE_DEVICES, so on this mixed 5090+3090 box one archs value applies to all
  lanes. The PR build must contain sm_86 AND sm_120 code to serve every lane.
- PITFALL: `cp -al` + json.dump(path, "w") writes THROUGH the hardlink and corrupts
  the source package's manifest. Always break the link (unlink + copy2) BEFORE editing.

Run ON the rig as al. Assumes ~/build/Strata @ branch f-rig-disk-cache built with
  cmake -DCMAKE_CUDA_ARCHITECTURES="86;120;120-virtual" (needed for mixed-GPU host)
Rollback: move the 0.1.56 dir out of components/engine, restart the lanes.
"""
import json, hashlib, os, shutil, subprocess, sys

BASE = "~/.local/share/strata/components/engine"
D55 = BASE + "/0.1.55-linux-x86_64-cuda-cuda13"
D56 = BASE + "/0.1.56-linux-x86_64-cuda-cuda13"
PR_BUILD = os.path.expanduser("~/build/Strata/build/strata")

if os.path.exists(D56):
    shutil.rmtree(D56)
subprocess.check_call(["cp", "-al", D55, D56])          # hardlinks, ~zero space
# break EVERY link we will modify, before modifying anything:
for rel in ("strata", "engine-component.json"):
    os.unlink(os.path.join(D56, rel))
shutil.copy2(PR_BUILD, D56 + "/strata")
shutil.copy2(D55 + "/engine-component.json", D56 + "/engine-component.json")

hh = hashlib.sha256()
with open(D56 + "/strata", "rb") as f:
    for c in iter(lambda: f.read(1 << 22), b""):
        hh.update(c)
j = json.load(open(D56 + "/engine-component.json"))
j["version"] = "0.1.56"
j["build_id"] = "pr3341-disk-cache+cuda13.0"
for e in j["files"]:
    if e["path"] == "strata":
        e["sha256"] = hh.hexdigest()
        e["bytes"] = os.path.getsize(D56 + "/strata")
j.pop("signature", None)                                # unsigned => accepted
json.dump(j, open(D56 + "/engine-component.json", "w"), indent=1)
print("built", D56)

sys.path.insert(0, "~/.local/share/strata/components/control/0.1.55")
from strata_control import engine_component as ec
from pathlib import Path
rep = ec.status(Path("~/.local/share/strata/components"))
print("usable:", rep["usable"], "->", (rep.get("engine") or {}).get("version"))
