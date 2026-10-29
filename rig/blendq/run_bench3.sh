#!/bin/bash
cd ~/blendq
PY=~/venv/ar/bin/python
$PY code_bench.py 18020 blend_mtp_3090 4 &
$PY code_bench.py 18021 prod38_mtp_3090 4 &
$PY code_bench.py 18022 blend_dflash_5090 4 &
wait; echo CODE3_DONE
#!/bin/bash
cd ~/blendq
PY=~/venv/ar/bin/python
$PY lcb_bench.py 18021 prod38_mtp_3090 4 &
$PY lcb_bench.py 18022 blend_dflash_5090 4 &
( until grep -q CODE3_DONE run_code3.log; do sleep 15; done; $PY lcb_bench.py 18020 blend_mtp_3090 4 ) &
wait; echo LCB3_DONE
