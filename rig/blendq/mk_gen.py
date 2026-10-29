#!/usr/bin/env python3
"""Build drafter/data/gen.jsonl (prompt_ids only) for capture_dflash2.py: RU chat + code + EN chat mix."""
import json, os, random, sys
from datasets import load_dataset
from transformers import AutoTokenizer

M = "~/HyperQwen-0.32/models/Qwen3.8-3.6-27B-blend-W4A16-AR"
OUT = "~/HyperQwen-0.32/drafter/data/gen.jsonl"
tok = AutoTokenizer.from_pretrained(M)
recs = []


def add(src, text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if 8 < len(ids) < 2048:
        recs.append({"id": f"{src}-{len(recs)}", "src": src, "think": True, "prompt_ids": list(ids)})


n = 0
for r in load_dataset("IlyaGusev/saiga_scored", split="train", streaming=True):
    if r.get("language") == "Russian" and (r.get("opus_score") or 0) >= 8:
        u = next((m["content"] for m in r["messages"] if m["role"] == "user"), None)
        if u:
            add("ru", u); n += 1
    if n >= 170:
        break
n = 0
for r in load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True):
    u = next((c["value"] for c in (r.get("conversations") or []) if c["from"] == "user"), None)
    if u and ("def " in u or "input" in u.lower() or "program" in u.lower()):
        add("code", u); n += 1
    if n >= 150:
        break
n = 0
for r in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True):
    add("en", r["messages"][0]["content"]); n += 1
    if n >= 80:
        break
random.Random(7).shuffle(recs)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
print(len(recs), {s: sum(r["src"] == s for r in recs) for s in ("ru", "code", "en")})
