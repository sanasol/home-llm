#!/bin/bash
# wait for rumix quant, flatten output dir, run prepare/ on it
cd ~/blendq
until grep -aq "QUANT_DONE" q3090a.log; do pgrep -f "ar_quant.py.*rumix" >/dev/null || { echo "rumix died"; exit 1; }; sleep 30; done
M=~/HyperQwen-0.32/models; S=$M/Qwen3.8-3.6-27B-blend; D=$S-W4A16-AR-rumix
sub=$(ls -d $D/*/ 2>/dev/null | head -1); [ -n "$sub" ] && mv $sub* $D/ && rmdir $sub
for f in video_preprocessor_config.json vocab.json merges.txt; do [ -f $D/$f ] || cp $S/$f $D/; done
cd ~/HyperQwen-0.32
for s in quant_lm_head quant_embed quant_mtp; do venv/bin/python prepare/$s.py $D; done
venv/bin/python prepare/build_draft_vocab.py $D --ids prepare/draft_vocab_ids.json
echo RUMIX_READY
