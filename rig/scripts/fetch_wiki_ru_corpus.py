#!/usr/bin/env python3
"""Fetch a bounded Russian corpus from aiplatforms/wikipedia-RuDataset
(cc-by-sa-3.0/gfdl) into a JSONL for build_draft_vocab.py --corpus.

NB: despite its name the dataset is chat pairs (columns: num, system, user,
assistant, u_tokens, a_tokens, u_lang, a_lang, cluster), not raw wiki text.
We take the assistant side of Russian rows (a_lang filter) - assistant prose
is closer to model output anyway, which is what the draft head scores.

Streams the parquet row groups; stops at --max-mb. Output lines are plain
{"text": ...} jsonl, which texts_from() reads directly.

Run on the rig with the HyperQwen venv python (has pyarrow + huggingface_hub):
  ~/HyperQwen-0.32/venv/bin/python fetch_wiki_ru_corpus.py --out ~/vocabcorpus/wiki_ru.jsonl
"""
import argparse, json, os, sys
from huggingface_hub import hf_hub_download

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="aiplatforms/wikipedia-RuDataset")
    ap.add_argument("--file", default="wikipedia_rudataset.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-mb", type=float, default=20.0)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--lang", default="ru", help="a_lang filter ('' to keep all)")
    ap.add_argument("--token", default=os.path.expanduser("~/.cache/huggingface/token"))
    a = ap.parse_args()

    tok = open(a.token).read().strip()
    path = hf_hub_download(a.repo, a.file, repo_type="dataset", token=tok)
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    print("row groups:", pf.num_row_groups, "rows:", pf.metadata.num_rows, flush=True)

    budget = int(a.max_mb * 1e6)
    written = 0; kept = 0; seen = 0
    langs = set()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)
            names = tbl.schema.names
            if "assistant" not in names:
                print("no 'assistant' column, have:", names); sys.exit(1)
            rows = tbl.to_pylist()
            for r in rows:
                seen += 1
                langs.add(r.get("a_lang"))
                if a.lang and r.get("a_lang") != a.lang:
                    continue
                t = r.get("assistant")
                if not isinstance(t, str) or len(t) < a.min_chars:
                    continue
                blob = json.dumps({"text": t}, ensure_ascii=False) + "\n"
                f.write(blob); written += len(blob.encode()); kept += 1
                if written >= budget:
                    break
            if written >= budget:
                break
    print(f"seen={seen} kept={kept} bytes={written} a_langs={sorted(x for x in langs if x)[:8]} -> {a.out}")

if __name__ == "__main__":
    main()
