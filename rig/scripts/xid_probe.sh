#!/bin/bash
n=$(journalctl -k --since "-15 min" 2>/dev/null | grep -c "NVRM: Xid")
{ echo '# HELP nvidia_xid_last15m NVIDIA Xid kernel events in last 15 min'
  echo '# TYPE nvidia_xid_last15m gauge'
  echo "nvidia_xid_last15m $n"; } > /var/lib/prometheus/node-exporter/xid.prom
