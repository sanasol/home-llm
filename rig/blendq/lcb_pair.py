#!/usr/bin/env python3
"""Paired comparison on common problem ids: lcb_pair.py A.jsonl B.jsonl"""
import json, statistics, sys


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p)) if "usage" in r}


a, b = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(set(a) & set(b))
ta = [a[i]["usage"]["completion_tokens"] for i in ids]
tb = [b[i]["usage"]["completion_tokens"] for i in ids]
fewer = sum(x < y for x, y in zip(ta, tb))
print(f"common={len(ids)}  pass {sys.argv[1]}={sum(a[i]['ok'] for i in ids)}  {sys.argv[2]}={sum(b[i]['ok'] for i in ids)}")
print(f"tokens sum: {sum(ta)} vs {sum(tb)}  ratio={sum(ta)/sum(tb):.2f}  median {statistics.median(ta):.0f} vs {statistics.median(tb):.0f}")
print(f"first uses fewer tokens on {fewer}/{len(ids)}; trunc {sum(a[i]['finish']!='stop' for i in ids)} vs {sum(b[i]['finish']!='stop' for i in ids)}")
both = [i for i in ids if a[i]["ok"] and b[i]["ok"]]
if both:
    print(f"on {len(both)} solved-by-both: tokens {sum(a[i]['usage']['completion_tokens'] for i in both)} vs "
          f"{sum(b[i]['usage']['completion_tokens'] for i in both)}")
print(f"wall sum (s): {sum(a[i]['wall'] for i in ids):.0f} vs {sum(b[i]['wall'] for i in ids):.0f}")
