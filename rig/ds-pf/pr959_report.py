#!/usr/bin/env python3
"""Markdown tables for the PR #959 comment from ~/ds-pf/pr959.jsonl (off vs on, per-run values + mean)."""
import json
import math
import statistics
from collections import defaultdict

rows = [json.loads(l) for l in open("~/ds-pf/pr959.jsonl") if l.strip()]
OFF, ON = "p-off", "p-on"


def mean(v):
    return statistics.mean(v) if v else float("nan")


def cell(v, fmt="{:.2f}"):
    return " / ".join(fmt.format(x) for x in v) if v else "-"


def pct(a, b):
    return f"{100 * (b - a) / a:+.0f}%" if a and not math.isnan(a) and not math.isnan(b) else "-"


g = defaultdict(list)
for r in rows:
    if r["kind"] in ("fresh", "agent-turn"):
        g[(r["kind"], r["chars"], r["arm"])].append(r)   # same text length = same request shape (token counts vary a little)
    elif r["kind"] == "agent-start":
        g[("start", 0, r["arm"])].append(r)

print("| prompt | without (`=0`), s | with (`=1`), s | change |\n| --- | ---: | ---: | ---: |")
keys = sorted({(k[0], k[1]) for k in g}, key=lambda k: ({"fresh": 0, "start": 1, "agent-turn": 2}[k[0]], k[1]))
for kind, size in keys:
    off = [x["read_s"] for x in g[(kind, size, OFF)]]
    on = [x["read_s"] for x in g[(kind, size, ON)]]
    toks = round(mean([x["read"] for x in g[(kind, size, OFF)] + g[(kind, size, ON)]]))
    label = {"fresh": f"{toks:,} tokens, fresh", "start": f"{toks:,}-token start, cold",
             "agent-turn": f"agent turn: {toks:,} new tokens"}[kind]
    print(f"| {label} | {cell(off)} | {cell(on)} | {pct(mean(off), mean(on))} |")

dec = {a: [r["dec"] for r in rows if r["kind"] == "decode" and r["arm"] == a] for a in (OFF, ON)}
print(f"\nDecode, 300-token answers, greedy (tok/s, every answer): without {cell(dec[OFF], '{:.1f}')}; "
      f"with {cell(dec[ON], '{:.1f}')} (means {mean(dec[OFF]):.1f} / {mean(dec[ON]):.1f}).")

# first answer token: top-10 log-probabilities, KL(off || on) over the tokens both lists hold, per prompt
lg = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["kind"] == "logits":
        lg[r["chars"]][r["arm"]].append(r)
print("\nFirst-token top-10, KL(without || with) over the common tokens, and the same for without vs without:")
for chars, by in sorted(lg.items()):
    def kl(p, q):
        common = set(p) & set(q)
        if not common:
            return float("nan")
        ps = {t: math.exp(p[t]) for t in common}
        qs = {t: math.exp(q[t]) for t in common}
        zp, zq = sum(ps.values()), sum(qs.values())
        return sum(ps[t] / zp * math.log((ps[t] / zp) / (qs[t] / zq)) for t in common)
    offs, ons = by.get(OFF, []), by.get(ON, [])
    cross = [kl(a["top"], b["top"]) for a in offs for b in ons]
    self_ = [kl(offs[0]["top"], offs[1]["top"])] if len(offs) > 1 else []
    firsts = {x["first"] for x in offs + ons}
    print(f"- {round(mean([x['read'] for x in offs + ons])):,} tokens: KL {cell(cross, '{:.1e}')}; "
          f"without vs without {cell(self_, '{:.1e}')}; first token the same in every run: {len(firsts) == 1}")

print("\n`DS4_PREFILL_STATS` lines (last run with the setting on):")
st = [r for r in rows if r["kind"] == "stats" and r["arm"] == ON]
for l in (st[-1]["lines"] if st else [])[-6:]:
    print("    " + l)
