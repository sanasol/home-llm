#!/bin/bash
# Switch the whole rig between FN (3 Strata lanes) and DS (Project Maya DeepSeek-V4.1-Flash, whole box).
# usage: rig_mode.sh fn|ds|status      (run on the rig as al; needs passwordless sudo for systemctl)
set -u
ARENA='mnt-strata\x2darena.mount'
health() { curl -sf -m 5 "http://127.0.0.1:$1/$2" >/dev/null; }
wait_for() { for _ in $(seq 1 "$3"); do health "$1" "$2" && return 0; sleep 5; done; return 1; }
status() {
  for u in llm-main llm-lane2 llm-lane3 maya-ds; do printf '%-10s %-9s %s\n' "$u" "$(systemctl is-active "$u")" "$(systemctl is-enabled "$u")"; done
}
case "${1:-status}" in
  ds)
    sudo systemctl stop llm-lane3 llm-lane2 llm-main "$ARENA"
    sudo systemctl disable -q llm-main llm-lane2 llm-lane3
    sudo systemctl enable -q maya-ds
    sudo systemctl start maya-ds
    wait_for 18080 v1/models 120 || { echo "maya-ds not ready"; exit 1; }
    ;;
  fn)
    sudo systemctl stop maya-ds
    sudo systemctl disable -q maya-ds
    sudo systemctl enable -q llm-main llm-lane2 llm-lane3
    sudo systemctl start "$ARENA" llm-main
    wait_for 8001 health 180 || { echo "llm-main not ready"; exit 1; }
    sudo systemctl start llm-lane2 llm-lane3
    wait_for 8003 health 120 && wait_for 8004 health 120 || { echo "3090 lanes not ready"; exit 1; }
    ;;
  status) ;;
  *) echo "usage: $0 fn|ds|status"; exit 2 ;;
esac
status
