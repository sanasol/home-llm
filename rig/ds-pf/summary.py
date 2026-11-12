#!/usr/bin/env python3
"""Per-arm medians from ~/ds-pf/results.jsonl: cold prefill t/s by size, agent-turn read seconds, decode t/s."""
import json
import statistics
from collections import defaultdict

rows = [json.loads(l) for l in open("~/ds-pf/results.jsonl") if l.strip()]
arms = list(dict.fromkeys(r["arm"] for r in rows))
cold_k = sorted({r["k"] for r in rows if r["kind"] == "cold"})
turns = sorted({r["add_k"] for r in rows if r["kind"] == "agent-turn"})
g = defaultdict(list)
for r in rows:
    if r["kind"] == "cold":
        g[(r["arm"], f"cold{r['k']}K t/s")].append(r["prefill"])
    elif r["kind"] == "agent-turn":
        g[(r["arm"], f"+{r['add_k']}K s")].append(r["read_s"])
    elif r["kind"] == "decode":
        g[(r["arm"], "decode t/s")].append(r["dec"])
cols = [f"cold{k}K t/s" for k in cold_k] + [f"+{k}K s" for k in turns] + ["decode t/s"]
med = lambda v: f"{statistics.median(v):.1f}" if v else "-"
print("arm".ljust(14) + "".join(c.rjust(14) for c in cols))
for a in arms:
    print(a.ljust(14) + "".join(med(g[(a, c)]).rjust(14) for c in cols))
