#!/usr/bin/env python3
"""Score RU harness jsonl files with the docs/17 glitch classifier. usage: score_ru3.py ru_a.jsonl ru_b.jsonl ..."""
import json, re, sys, statistics
CJK = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+')
CYR = 'а-яА-ЯёЁ'
WORD = re.compile(r'[\w\u0400-\u04ff]+')
HOMO_LAT = set('aceopxyABCEHKMOPTXy')
SLAV = re.compile(r'[čćšžđłąęśźżńó]', re.I)
FENCE = re.compile(r'```.*?```|`[^`\n]*`', re.S)


def glitches(txt):
    prose = FENCE.sub(' ', txt or '')
    out = []
    out += [('cjk', m.group()) for m in CJK.finditer(prose)]
    for m in WORD.finditer(prose):
        w = m.group()
        has_c = re.search('[' + CYR + ']', w)
        lat = re.findall('[a-zA-Z]', w)
        if has_c and lat:
            out.append(('homoglyph' if all(c in HOMO_LAT for c in lat) and len(lat) <= 2 else 'blend', w))
        elif has_c and SLAV.search(w):
            out.append(('slavic', w))
    return out


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    ok = [r for r in rows if 'usage' in r]
    g = [glitches(r['content']) for r in ok]
    n = len(ok)
    bad = sum(1 for x in g if x)
    cjk = sum(1 for x in g if any(k == 'cjk' for k, _ in x))
    ct = [r['usage']['completion_tokens'] for r in ok]
    tps = [r['usage']['completion_tokens'] / r['wall'] for r in ok]
    rlen = [len(r.get('reasoning') or '') for r in ok]
    clen = [len(r['content']) for r in ok]
    trunc = sum(1 for r in ok if r.get('finish') != 'stop')
    ex = [w for x in g for _, w in x][:12]
    print(f"{path}: n={n} errors={len(rows)-n} glitch={bad/n:.1%} cjk={cjk/n:.1%} trunc={trunc} "
          f"med_out_tok={statistics.median(ct):.0f} med_reasoning_chars={statistics.median(rlen):.0f} "
          f"med_answer_chars={statistics.median(clen):.0f} med_tps={statistics.median(tps):.0f} ex={ex}")
