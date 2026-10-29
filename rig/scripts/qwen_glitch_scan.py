#!/usr/bin/env python3
"""Scan Hermes state.db for Qwen output glitches (CJK leaks, Latin look-alike letters in
Cyrillic words, half-English blends, Serbian/Polish leakage) per engine lane.
Usage: qwen_glitch_scan.py out.json   (reads /root/.hermes/state.db read-only). docs/17."""
import sqlite3, re, json, collections, sys, hashlib
c = sqlite3.connect('file:/root/.hermes/state.db?mode=ro', uri=True)
OJ_START = 1793238503  # 2026-10-28 21:48:23 EDT (OpenJet rollout commit); A back on vLLM 2026-10-29 13:38 EDT
CJK = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+')
CYR = 'а-яА-ЯёЁ'
WORD = re.compile(r'[\w\u0400-\u04ff]+')
HOMO_LAT = set('aceopxyABCEHKMOPTXy')  # latin look-alikes of cyrillic letters
SLAV = re.compile(r'[čćšžđłąęśźżńó]', re.I)
FENCE = re.compile(r'```.*?```|`[^`\n]*`', re.S)


def lane(model, ts):
    if model == 'qwen3.8-27b-a':
        return '27b-a/openjet' if ts >= OJ_START else '27b-a/vllm'
    if model in ('qwen3.8-flash-next-iq3_s', 'qwen3.8-flash-next'):
        return 'flash-next-iq3_s'
    if model in ('qwen3.8-27b', 'qwen3.8-27b-b'):  # historical names stay readable in old sessions
        return '27b-vllm(B/legacy)'
    return 'other:' + model


def classify(w):
    has_c = re.search('[' + CYR + ']', w)
    lat = re.findall('[a-zA-Z]', w)
    if not (has_c and lat):
        return None
    if all(ch in HOMO_LAT for ch in lat) and len(lat) <= 2:
        return 'homoglyph'
    return 'blend'


seen = set()
stats = collections.defaultdict(collections.Counter)
ex = collections.defaultdict(list)
for sid, model in c.execute("select id, model from sessions where model like 'qwen3.8%' or model like 'Qwen3.8%'").fetchall():
    for content, rc, ts in c.execute("select content, reasoning_content, timestamp from messages where session_id=? and role='assistant'", (sid,)):
        L = lane(model, ts)
        for field, txt in (('content', content), ('reasoning', rc)):
            if not txt or len(txt) < 20:
                continue
            h = hashlib.md5(txt.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            prose = FENCE.sub(' ', txt)
            st = stats[(L, field)]
            st['chars'] += len(prose)
            st['msgs'] += 1
            ru = len(re.findall('[' + CYR + ']', prose)) / max(1, len(prose)) > 0.2
            st['ru_msgs'] += ru
            for m in CJK.finditer(prose):
                st['cjk'] += 1
                if len(ex[(L, field, 'cjk')]) < 25:
                    ex[(L, field, 'cjk')].append(prose[max(0, m.start() - 50):m.end() + 40].replace('\n', ' '))
            for m in WORD.finditer(prose):
                w = m.group()
                k = classify(w)
                if k:
                    st[k] += 1
                    if len(ex[(L, field, k)]) < 25:
                        ex[(L, field, k)].append(w)
                elif re.search('[' + CYR + ']', w) and SLAV.search(w):
                    st['slavic'] += 1
                    ex[(L, field, 'slavic')].append(w)
                elif SLAV.search(w) and ru and not re.search('[' + CYR + ']', w):
                    st['slavic_latin'] += 1
                    if len(ex[(L, field, 'slavic_latin')]) < 25:
                        ex[(L, field, 'slavic_latin')].append(w)
rows = []
for (L, f), st in sorted(stats.items()):
    per = 1e5 / max(1, st['chars'])
    rows.append(dict(lane=L, field=f, msgs=st['msgs'], ru_msgs=st['ru_msgs'], kchars=round(st['chars'] / 1000),
                     cjk_per100k=round(st['cjk'] * per, 2), homoglyph_per100k=round(st['homoglyph'] * per, 2),
                     blend_per100k=round(st['blend'] * per, 2), slavic_per100k=round((st['slavic'] + st['slavic_latin']) * per, 2),
                     raw={k: st[k] for k in ('cjk', 'homoglyph', 'blend', 'slavic', 'slavic_latin')}))
json.dump({'rows': rows, 'ex': {'|'.join(k): v[:25] for k, v in ex.items()}}, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
for r in rows:
    print(r)
