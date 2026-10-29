#!/bin/bash
# RU harness on 3 engines in parallel: a=blend AR (18020), b=prod 3.8 (18021), c=blend AR-rumix (5090, 18022)
cd ~/blendq
K=~/HyperQwen/api_key.txt
mtp() { curl -s localhost:$1/metrics -H "Authorization: Bearer $(cat $K)" | grep -E "^vllm:spec_decode_num_(accepted|draft)_tokens_total" | awk '{print $2}' | tr '\n' ' '; }
for p in 18020 18021 18022; do echo "MTP0 $p $(mtp $p)"; done > mtp.txt
rm -f ru_a.jsonl ru_b.jsonl ru_c.jsonl
for e in a b c; do /usr/bin/python3 ru_ab3.py $e new 120 3 ru_$e.jsonl & done
wait
for p in 18020 18021 18022; do echo "MTP1 $p $(mtp $p)"; done >> mtp.txt
echo RU3_DONE
