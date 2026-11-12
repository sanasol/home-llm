#!/usr/bin/env python3
"""Round 2 arms (12.11 late): everything on top of the CURRENT production config (maya-ds4.json: PR #959 build,
COPY_DMA + AI_RUN + YOCO_FLOW + PREFILL_CPU, split 18,9,13). One knob group per arm.
  r-prod     production as is (the baseline of this round)
  r-dec      + decode launch/GEMV work, bit-exact: Q8MV_TC, HC_FUSE=2, ROUTE_FUSE=2, ROUTER_BF16
  r-dec-dc   + the same + DRAFT_COST (DSpark window sized by a verified row's cost)
  r-s1499    production + split 14,9,17 (layers 0-19 = the skip's work: more of them on the 5090 and x8 3090)
  r-s2010    production + split 20,8,12
  r-s1688    production + split 16,8,16
"""
import copy, json
LIVE = "~/maya/maya-ds4.json"
base = json.load(open(LIVE))
DEC = {"MAYA_DS4_Q8MV_TC": "1", "DS4_HC_FUSE": "2", "DS4_ROUTE_FUSE": "2", "DS4_ROUTER_BF16": "1"}
ARMS = {
    "r-prod": {},
    "r-dec": DEC,
    "r-dec-dc": {**DEC, "MAYA_DS4_DRAFT_COST": "1"},
    "r-s1499": {"DS4_SPLIT": "14,9,17"},
    "r-s2010": {"DS4_SPLIT": "20,8,12"},
    "r-s1688": {"DS4_SPLIT": "16,8,16"},
}
for name, env in ARMS.items():
    c = copy.deepcopy(base)
    c["env"] = {**base["env"], **env}
    c["log"] = f"~/ds-pf/{name}.engine.log"
    json.dump(c, open(f"~/ds-pf/{name}.json", "w"), indent=1)
    print(name, env)
