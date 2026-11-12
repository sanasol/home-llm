#!/bin/bash
# PR #959 A/B: the same binary (v1.1.61 + #959) with MAYA_DS4_PREFILL_CPU=0 vs =1, nothing else different.
# Fresh engine per run, one frozen expert profile restored before every start, order off/on/off/on.
# Stops maya-ds for the run, restores it (and its profile) on exit. Results: ~/ds-pf/pr959.jsonl
set -u
D=~/ds-pf
M=~/maya
ORDER=${ORDER:-"p-off p-on p-off p-on"}
export PATH=/usr/local/cuda/bin:$PATH CPATH=$M/.cuda-headers/nvidia/cu13/include
log() { echo "[$(date +%T)] $*"; }

busy=$(curl -s -m 5 http://127.0.0.1:18080/metrics | python3 -c "import json,sys;l=json.load(sys.stdin)['live'];print(l['state'], l.get('queued') or 0)" 2>/dev/null)
case "$busy" in idle\ 0|"") ;; *) log "DS is busy ($busy)"; exit 1;; esac
systemctl is-active -q rig-mode-switch && { log "a rig mode switch is running"; exit 1; }

PACK=$(python3 -c "import json;a=json.load(open('$M/maya-ds4.json'))['args'];print(a[a.index('--pack')+1])")
log "stopping maya-ds"
sudo -n systemctl stop maya-ds
cp "$PACK/expert-profile.bin" $D/pr959-profile.frozen
cp "$PACK/expert-profile.bin" $D/pr959-profile.live-backup
trap 'log "restoring maya-ds"; pkill -INT -f "maya.py --config $D/p-"; sleep 15; cp $D/pr959-profile.live-backup "$PACK/expert-profile.bin"; sudo -n systemctl start maya-ds' EXIT

r=0
for arm in $ORDER; do
  r=$((r + 1))
  log "run $r: $arm"
  cp $D/pr959-profile.frozen "$PACK/expert-profile.bin"
  : > $D/$arm.engine.log
  (cd $M && setsid ./.venv/bin/python maya.py --config $D/$arm.json --port 18080 --plain > $D/$arm.server.log 2>&1 &)
  for i in $(seq 1 90); do curl -sf -m 3 http://127.0.0.1:18080/v1/models | grep -q '"id"' && break; sleep 5; done
  python3 $D/pr959_probe.py "$arm" "$r" | tee -a $D/pr959.jsonl
  cp $D/$arm.engine.log $D/pr959-run$r-$arm.engine.log
  pkill -INT -f "maya.py --config $D/$arm.json"
  for i in $(seq 1 40); do pgrep -f "maya.py --config $D/$arm.json" >/dev/null || break; sleep 2; done
  pkill -9 -f "maya.py --config $D/$arm.json" 2>/dev/null; sleep 3
done
python3 $D/pr959_report.py
