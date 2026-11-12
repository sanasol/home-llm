#!/bin/bash
# DS prefill A/B on the rig. Stops the live maya-ds (with the switch's lock: rig-mode-switch must be idle),
# runs each arm as its own Maya on :18080 with ~/ds-pf/<arm>.json, then restarts maya-ds as it was.
# usage: run_ab.sh [ARM ...]      (default: every arm, two rounds A..H A..H)
#        COLD=24,128 run_ab.sh a-base f-pcpu      (shorter prompts for a quick pass)
# Results: ~/ds-pf/results.jsonl (one line per measurement) + summary.py.
set -u
D=~/ds-pf
M=~/maya
COLD=${COLD:-24,128,400}
ROUNDS=${ROUNDS:-2}
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(a-base b-copy c-yoco d-yoco1499 e-yoco2010 f-pcpu g-pcpu-em12 h-norestore)
export PATH=/usr/local/cuda/bin:$PATH CPATH=$M/.cuda-headers/nvidia/cu13/include
log() { echo "[$(date +%T)] $*"; }

busy=$(curl -s -m 5 http://127.0.0.1:18080/metrics | python3 -c "import json,sys;l=json.load(sys.stdin)['live'];print(l['state'], l.get('queued') or 0)" 2>/dev/null)
case "$busy" in idle\ 0|"") ;; *) log "DS is busy ($busy): stop your sessions first"; exit 1;; esac
systemctl is-active -q rig-mode-switch && { log "a rig mode switch is running"; exit 1; }
[ -x ~/maya-pf/build-pf/maya-ds4 ] || log "warning: build-pf missing, the f/g/h arms will fail"

log "stopping maya-ds (its conversations go to ~/ds-slots)"
sudo -n systemctl stop maya-ds
PACK=$(python3 -c "import json;a=json.load(open('$M/maya-ds4.json'))['args'];print(a[a.index('--pack')+1])")
# every arm starts from the expert profile the live server saved (each run saves its own at stop: freeze it)
cp "$PACK/expert-profile.bin" $D/expert-profile.frozen
cp "$PACK/expert-profile.bin" $D/expert-profile.live-backup
trap 'log "restoring maya-ds"; pkill -INT -f "maya.py --config $D/"; sleep 15; cp $D/expert-profile.live-backup "$PACK/expert-profile.bin"; sudo -n systemctl start maya-ds' EXIT

for round in $(seq 1 "$ROUNDS"); do
  for arm in "${ARMS[@]}"; do
    log "round $round: $arm"
    # each arm starts with the same frozen expert profile (the server saves its own at each stop)
    cp $D/expert-profile.frozen "$PACK/expert-profile.bin"
    : > $D/$arm.engine.log
    (cd $M && setsid ./.venv/bin/python maya.py --config $D/$arm.json --port 18080 --plain > $D/$arm.server.log 2>&1 &)
    for i in $(seq 1 90); do curl -sf -m 3 http://127.0.0.1:18080/v1/models | grep -q '"id"' && break; sleep 5; done
    python3 $D/probe.py "$arm" --cold "$COLD" | sed "s/^{/{\"round\": $round, /" | tee -a $D/results.jsonl
    grep -h "DS4_PREFILL_CPU gpu=" $D/$arm.engine.log | tail -3
    pkill -INT -f "maya.py --config $D/$arm.json"
    for i in $(seq 1 40); do pgrep -f "maya.py --config $D/$arm.json" >/dev/null || break; sleep 2; done
    pkill -9 -f "maya.py --config $D/$arm.json" 2>/dev/null; sleep 3
  done
done
python3 $D/summary.py
