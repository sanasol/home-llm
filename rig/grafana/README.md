# Grafana/Prometheus provisioning (rig)

Dashboards `rig-overview` and `rig-litellm` are GENERATED — edit `build_dashboards.py`, then:

  python3 rig/grafana/build_dashboards.py          # writes dashboards/*.json

Deploy (idempotent, from repo root; scp to the rig is dead from hub -> ssh tee):

  R=user@rig
  ssh $R 'sudo -n tee /etc/grafana/provisioning/datasources/rig.yaml >/dev/null' < rig/grafana/provisioning-datasources.yaml
  ssh $R 'sudo -n tee /etc/grafana/provisioning/dashboards/rig.yaml  >/dev/null' < rig/grafana/provisioning-dashboards.yaml
  ssh $R 'sudo -n tee /etc/grafana/provisioning/alerting/rig.yaml    >/dev/null' < rig/grafana/provisioning-alerting.yaml
  for f in rig-overview rig-litellm; do
    ssh $R "sudo -n tee /var/lib/grafana/dashboards/$f.json >/dev/null" < rig/grafana/dashboards/$f.json; done
  ssh $R 'sudo chown grafana:grafana /var/lib/grafana/dashboards/*.json; sudo systemctl restart grafana-server'

Foreign dashboards (node-full = grafana.com 1860, anything imported) MUST pass the tooltip
rule before deploy:

  ssh $R 'sudo cat /var/lib/grafana/dashboards/node-full.json' > /tmp/nf.json
  python3 rig/grafana/build_dashboards.py --enforce /tmp/nf.json
  ssh $R 'sudo -n tee /var/lib/grafana/dashboards/node-full.json >/dev/null' < /tmp/nf.json

House rules (docs/14 "Правила дашбордов"): every timeseries tooltip mode=multi ("All"),
sort desc, dashboard graphTooltip=1; GPU names via `rig_gpu_info` join, LLM series from
`rig_llm_*`; after deploy run every expr through Prometheus query_range — empty = bug.

Prometheus config = rig/config/prometheus.yml -> /etc/prometheus/prometheus.yml; validate with
`/opt/prometheus/bin/promtool check config`, apply with `sudo systemctl kill -s HUP prometheus`
(no reload verb in the unit, no lifecycle API).

LLM exporter: rig/scripts/rig_llm_exporter.py -> /opt/rig/bin/, unit rig/config/rig-llm-exporter.service.

node-full dashboard = grafana.com id 1860 with datasource patched to uid `rigprom`
(patch script: rewrite all datasource uids + drop __inputs; file lives on rig
/var/lib/grafana/dashboards/node-full.json).

Grafana admin: user `admin`, password file /opt/rig/grafana_admin.txt (root,600).
Gotchas: execErrState enum is `OK` not `Ok`; alert rules fail whole-file provisioning
on parse error; v13 CLI: `grafana cli admin reset-admin-password`; prometheus token
files /etc/prometheus/tokens/*.key (root:prometheus 640); gpu exporter metric names are
nvidia_smi_* (temperature_gpu, power_draw_watts), NOT temperature_c/power_draw_w;
vLLM 0.32 renamed gpu_cache_usage_perc -> kv_cache_usage_perc; external_prefix_cache_* is
the KV-connector counter (always 0 here) — real prefix cache is prefix_cache_hits/queries;
Strata /metrics is custom JSON -> Prometheus only sees `up` (tok/s via rig_llm_exporter).
