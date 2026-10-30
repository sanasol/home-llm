#!/bin/bash
# KV dtype and DFlash2 draft-window sweep on the Deck. Stops ninfer.service, serves each config on
# 127.0.0.1:18080 (no key, no Hermes traffic), runs bench.py (+ needle.py at ~160K for the KV configs),
# then starts ninfer.service again. Results land in ~/exp/<name>.*
set -u
cd ~/ninfer
OUT=~/exp; mkdir -p $OUT
URL=http://127.0.0.1:18080
COMMON="--host 127.0.0.1 --port 18080 --kv-capacity auto --max-concurrency 2 --device-state-slots 2
        --lm-head-draft --preserve-thinking --vision --host-context-mib 4096 --default-max-tokens 32768"

serve() {  # name, extra flags...
  local name=$1; shift
  ./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer $COMMON "$@" > $OUT/$name.log 2>&1 &
  SPID=$!
  for i in $(seq 1 120); do
    curl -sf $URL/health >/dev/null && break
    kill -0 $SPID 2>/dev/null || { echo "$name: server died"; tail -5 $OUT/$name.log; return 1; }
    sleep 5
  done
  grep -E "capacity \| KV" $OUT/$name.log | tail -1
}

stop() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 3; }

fits() {  # planner-only check: does this flag set get past "planning runtime"?
  timeout 6 ./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer $COMMON "$@" 2>&1 | grep -q "loading weights"
}

run() {  # name, needle(yes/no), flags...
  local name=$1 needle=$2; shift 2
  echo "=== $name: $*"
  serve $name "$@" || return
  python3 ~/bench.py $URL qwen3.8-27b $OUT/$name.bench.json ~/qwen38-27b-vision-rtx-5090/engine/zig/src
  [ "$needle" = yes ] && python3 ~/needle.py $URL qwen3.8-27b $OUT/$name.needle.json 160000 ~/ninfer/src ~/qwen38-27b-vision-rtx-5090/engine/zig/src
  stop
}

sudo systemctl stop ninfer
run fp8-k7b  no --kv-dtype fp8  --spec dflash2 --draft-tokens 7 --max-context 212992
echo "=== k8v4-k7-240k"
serve k8v4-k7-240k --kv-dtype k8v4 --spec dflash2 --draft-tokens 7 --max-context 240000 && {
  python3 ~/bench.py $URL qwen3.8-27b $OUT/k8v4-k7-240k.bench.json ~/qwen38-27b-vision-rtx-5090/engine/zig/src
  python3 ~/needle.py $URL qwen3.8-27b $OUT/k8v4-k7-240k.needle.json 200000 ~/ninfer/src ~/qwen38-27b-vision-rtx-5090/engine/zig/src
  stop; }
sudo systemctl start ninfer
echo "=== done; ninfer.service started"
