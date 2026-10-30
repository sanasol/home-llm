#!/usr/bin/env python3
"""Held-out corpus coverage for draft-vocab id lists.

Replicates HyperQwen prepare/build_draft_vocab.py methodology exactly:
same tokenizer, same jsonl texts_from() field handling, every 10th text held
out, same SCRIPTS unicode ranges and classify_vocab logic (copied from the
tool because that module exits on import without full argv).

Reports on the held-out split, per id list:
  - overall % of held-out corpus tokens the list can draft
  - per-script coverage of the corpus tokens belonging to that script

Run on the rig:
  ~/HyperQwen-0.32/venv/bin/python draft_vocab_coverage.py \
    --model /models/qwen3.8-27b-gptq-lean \
    --corpus ~/vocabcorpus/wiki_ru.jsonl ~/ru_prompts.jsonl \
    --lists ~/HyperQwen-0.32/prepare/draft_vocab_ids.json \
            ~/vocabcorpus/draft_vocab_ids_cyrillic_pub.json
"""
import argparse, json, collections

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--corpus", nargs="+", required=True)
ap.add_argument("--lists", nargs="+", required=True)
a = ap.parse_args()

# ---- copied verbatim from prepare/build_draft_vocab.py ----
SCRIPTS = {
    "han":       [(0x3400,0x4DBF),(0x4E00,0x9FFF),(0xF900,0xFAFF),(0x20000,0x2FA1F)],
    "kana":      [(0x3040,0x30FF),(0x31F0,0x31FF),(0xFF66,0xFF9F)],
    "hangul":    [(0x1100,0x11FF),(0x3130,0x318F),(0xAC00,0xD7AF)],
    "cjk_punct": [(0x3000,0x303F),(0xFF00,0xFF65),(0xFFA0,0xFFEF)],
    "cyrillic":  [(0x0400,0x04FF),(0x0500,0x052F),(0x1C80,0x1C8F),(0x2DE0,0x2DFF),(0xA640,0xA69F)],
    "latin_ext": [(0x00C0,0x024F),(0x1E00,0x1EFF),(0x2C60,0x2C7F),(0xA720,0xA7FF)],
}

def scripts_of(ch):
    c = ord(ch)
    return {n for n, rs in SCRIPTS.items() if any(a <= c <= b for a, b in rs)}

def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff"))))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))

def classify_vocab(model_dir):
    uni2byte = {v: k for k, v in bytes_to_unicode().items()}
    vocab = json.load(open(model_dir + "/tokenizer.json"))["model"]["vocab"]
    n = max(vocab.values()) + 1
    whole = bytearray(n)
    kindset = [frozenset()] * n
    nfrag = 0
    for piece, i in vocab.items():
        try:
            raw = bytes(uni2byte[c] for c in piece)
        except KeyError:
            raw = piece.encode("utf-8", errors="ignore")
        if not raw:
            continue
        try:
            txt = raw.decode("utf-8")
        except UnicodeDecodeError:
            nfrag += 1
            continue
        if not txt:
            continue
        whole[i] = 1
        s = set()
        for ch in txt:
            s |= scripts_of(ch)
        kindset[i] = frozenset(s)
    return whole, kindset, nfrag
# ---- end copied block ----

def texts_from(path, limit_bytes=20_000_000):
    # exact copy of the tool's jsonl branch
    n = 0
    for line in open(path):
        try: r = json.loads(line)
        except Exception: continue
        parts = []
        for k in ("prompt", "response", "text"):
            if isinstance(r.get(k), str): parts.append(r[k])
        if isinstance(r.get("messages"), list):
            parts += [m.get("content", "") for m in r["messages"] if isinstance(m.get("content"), str)]
        t = "\n".join(parts); yield t; n += len(t)
        if n > limit_bytes: return

from transformers import AutoTokenizer
d = a.model.rstrip("/") + "/"
tok = AutoTokenizer.from_pretrained(d)
whole, kindset, nfrag = classify_vocab(d)
nrows = len(whole)
lists = {p.rsplit("/", 1)[-1]: set(json.load(open(p))) for p in a.lists}

for C in a.corpus:
    held = collections.Counter()
    ntexts = 0
    for j, t in enumerate(texts_from(C)):
        ntexts += 1
        if j % 10 == 0:
            held.update(tok(t, add_special_tokens=False).input_ids)
    tot = sum(held.values())
    print(f"\n== {C}: texts {ntexts}, held-out tokens {tot}")
    for name, ids in lists.items():
        cov = sum(c for i, c in held.items() if i in ids)
        line = f"  {name}: overall {100*cov/max(1,tot):.2f}%"
        for scr in ("cyrillic", "han", "latin_ext"):
            stot = sum(c for i, c in held.items() if scr in kindset[i])
            sin = sum(c for i, c in held.items() if scr in kindset[i] and i in ids)
            if stot:
                line += f" | {scr} {100*sin/stot:.2f}% of {stot}"
        print(line)
