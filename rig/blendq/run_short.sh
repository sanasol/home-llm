#!/bin/bash
cd ~/blendq
PY=~/venv/ar/bin/python
export DIFF=hard LIMIT=30
$PY lcb_bench.py 18020 s_blend_df4 5 & $PY lcb_bench.py 18021 s_prod38 5 & wait
K=~/HyperQwen/api_key.txt
mtp() { curl -s localhost:$1/metrics -H "Authorization: Bearer $(cat $K)" | grep -E "^vllm:spec_decode_num_(accepted|draft)_tokens_total" | awk "{print \$2}" | tr "\n" " "; }
for p in 18020 18021; do echo "RU0 $p $(mtp $p)"; done > mtp_short.txt
rm -f ru_sa.jsonl ru_sb.jsonl
/usr/bin/python3 ru_ab3.py a new 30 5 ru_sa.jsonl & /usr/bin/python3 ru_ab3.py b new 30 5 ru_sb.jsonl & wait
for p in 18020 18021; do echo "RU1 $p $(mtp $p)"; done >> mtp_short.txt
echo SHORT_DONE
