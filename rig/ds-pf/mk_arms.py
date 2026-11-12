#!/usr/bin/env python3
"""Write one Maya config per A/B arm from the live config (maya-ds4.json): same model/context/split/slots,
only the engine binary and the prefill env differ. Arms run on :18080 one at a time (the box holds one DS)."""
import copy, json
LIVE = "~/maya/maya-ds4.json"
PF_EXE = "~/maya-pf/build-pf/maya-ds4"          # v1.1.61 + PR #959
base = json.load(open(LIVE))
COPY = {"DS4_FLOW_COPY_DMA": "1", "DS4_AI_RUN": "1"}                 # bit-exact
YOCO = {"DS4_YOCO_FLOW": "1"}                                        # skip layers 20-39 on the split (last bits move)
PCPU = {"MAYA_DS4_PREFILL_CPU": "1", "DS4_PREFILL_STATS": "1"}       # PR #959
ARMS = {
    "a-base":      (base["exe"], {}),
    "b-copy":      (base["exe"], COPY),
    "c-yoco":      (base["exe"], {**COPY, **YOCO}),
    "d-yoco1499":  (base["exe"], {**COPY, **YOCO, "DS4_SPLIT": "14,9,17"}),
    "e-yoco2010":  (base["exe"], {**COPY, **YOCO, "DS4_SPLIT": "20,8,12"}),
    "f-pcpu":      (PF_EXE,      {**COPY, **YOCO, **PCPU}),
    "g-pcpu-em12": (PF_EXE,      {**COPY, **YOCO, **PCPU, "MAYA_DS4_PREFILL_CPU_EXPERT_MS": "1.2"}),
    "h-norestore": (PF_EXE,      {**COPY, **YOCO, **PCPU, "MAYA_DS4_RESTORE_POINTS": "0"}),
}
for name, (exe, env) in ARMS.items():
    c = copy.deepcopy(base)
    c["exe"] = exe
    c["env"] = {**base.get("env", {}), **env}
    c["log"] = f"~/ds-pf/{name}.engine.log"
    json.dump(c, open(f"~/ds-pf/{name}.json", "w"), indent=1)
    print(name, exe.split("/")[-2], {k: v for k, v in c["env"].items() if k not in base.get("env", {}) or base["env"][k] != v})
