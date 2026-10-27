# Grafana/Prometheus provisioning (rig)

Deploy (idempotent, from repo root):
  scp rig/grafana/provisioning-datasources.yaml user@rig:/tmp/graf_ds.yaml
  scp rig/grafana/provisioning-dashboards.yaml  user@rig:/tmp/graf_dash.yaml
  scp rig/grafana/provisioning-alerting.yaml    user@rig:/tmp/graf_alert.yaml
  scp rig/grafana/dashboards/rig-overview.json  user@rig:/tmp/graf_ov.json
  ssh user@rig 'sudo cp /tmp/graf_ds.yaml /etc/grafana/provisioning/datasources/rig.yaml
              sudo cp /tmp/graf_dash.yaml /etc/grafana/provisioning/dashboards/rig.yaml
              sudo cp /tmp/graf_alert.yaml /etc/grafana/provisioning/alerting/rig.yaml
              sudo cp /tmp/graf_ov.json /var/lib/grafana/dashboards/
              sudo systemctl restart grafana-server'

node-full dashboard = grafana.com id 1860 with datasource patched to uid `rigprom`
(patch script: rewrite all datasource uids + drop __inputs; file lives on rig
/var/lib/grafana/dashboards/node-full.json).

Grafana admin: user `admin`, password file /opt/rig/grafana_admin.txt (root,600).
Gotchas: execErrState enum is `OK` not `Ok`; alert rules fail whole-file provisioning
on parse error; v13 CLI: `grafana cli admin reset-admin-password`; prometheus token
files /etc/prometheus/tokens/*.key (root:prometheus 640); gpu exporter metric names are
nvidia_smi_* (temperature_gpu, power_draw_watts), NOT temperature_c/power_draw_w;
Strata /metrics is custom JSON -> Prometheus only sees `up`.
