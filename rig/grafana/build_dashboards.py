#!/usr/bin/env python3
"""Generate rig Grafana dashboards (rig-overview, rig-litellm) and enforce house rules.

    python3 rig/grafana/build_dashboards.py            # rewrite dashboards/*.json from code
    python3 rig/grafana/build_dashboards.py --enforce FILE.json [...]   # patch foreign dashboards
                                                       # (e.g. node-full.json pulled from the rig)

House rules (docs/14, "Grafana dashboard rules"), applied by enforce() to EVERY dashboard:
  * every timeseries/graph-like panel: tooltip mode "multi" (UI: "All"), sorted desc;
  * dashboard graphTooltip = 1 (shared crosshair across panels);
  * GPU series are named via `* on(uuid) group_left(gpu) rig_gpu_info`, never raw uuids;
  * LLM series come from rig_llm_* (one family covers the three Strata lanes FN 5090 /
    FN 3090 #1 / FN 3090 #2), legend "{{lane}}" (= human lane name; GPU panels use {{gpu}}).
Edit THIS file, never the generated JSON.
"""
import json
import sys
from pathlib import Path

DS = {"type": "prometheus", "uid": "rigprom"}
OUT = Path(__file__).resolve().parent / "dashboards"
TOOLTIP = {"mode": "multi", "sort": "desc", "hideZeros": False}
LEGEND = {"displayMode": "list", "placement": "bottom", "showLegend": True}
LLM = "{{lane}}"
GPU_JOIN = " * on(uuid) group_left(gpu) rig_gpu_info"
LANES = (("0", "FN 5090"), ("1", "FN 3090 #1"), ("2", "FN 3090 #2"))   # nvidia-smi index -> lane


def enforce(d):
    """Apply the tooltip house rule to every panel (recursing into rows)."""
    d["graphTooltip"] = 1

    def walk(panels):
        for p in panels or []:
            if p.get("type") == "row":
                walk(p.get("panels"))
                continue
            opts = p.setdefault("options", {})
            if p.get("type") in ("timeseries", "barchart", "trend", "xychart", "histogram",
                                 "state-timeline", "status-history", "candlestick", "heatmap"):
                opts["tooltip"] = dict(TOOLTIP)
            elif p.get("type") == "graph":  # legacy panel (node-full) -> shared sorted tooltip
                p["tooltip"] = {"shared": True, "sort": 2, "value_type": "individual"}
    walk(d.get("panels"))
    return d


class Board:
    def __init__(self):
        self.panels, self.x, self.y, self.h, self.pid = [], 0, 0, 0, 1

    def row(self, title):
        if self.x:
            self.y += self.h
            self.x = self.h = 0
        self.panels.append({"type": "row", "title": title, "collapsed": False, "id": self.pid,
                            "gridPos": {"h": 1, "w": 24, "x": 0, "y": self.y}, "panels": []})
        self.pid += 1
        self.y += 1

    def _place(self, w, h):
        if self.x + w > 24:
            self.y += self.h
            self.x = self.h = 0
        gp = {"h": h, "w": w, "x": self.x, "y": self.y}
        self.x += w
        self.h = max(self.h, h)
        return gp

    def ts(self, title, targets, unit="none", w=8, h=8, desc=None, minv=None, maxv=None,
           stack=False, fill=10, overrides=None, decimals=None, draw="line"):
        dfl = {"unit": unit, "custom": {"drawStyle": draw, "lineWidth": 2, "fillOpacity": fill,
                                        "showPoints": "never", "spanNulls": 60000,
                                        "stacking": {"mode": "normal" if stack else "none"}}}
        if minv is not None:
            dfl["min"] = minv
        if maxv is not None:
            dfl["max"] = maxv
        if decimals is not None:
            dfl["decimals"] = decimals
        p = {"id": self.pid, "title": title, "type": "timeseries", "datasource": DS,
             "gridPos": self._place(w, h), "targets": self._t(targets),
             "fieldConfig": {"defaults": dfl, "overrides": overrides or []},
             "options": {"tooltip": dict(TOOLTIP), "legend": dict(LEGEND)}}
        if desc:
            p["description"] = desc
        self.pid += 1
        self.panels.append(p)
        return p

    def stat(self, title, targets, unit="none", w=4, h=5, desc=None, decimals=1, thresholds=None,
             color_mode="value", text_mode="value_and_name", time_from=None, graph="area"):
        p = {"id": self.pid, "title": title, "type": "stat", "datasource": DS,
             "gridPos": self._place(w, h), "targets": self._t(targets),
             "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals,
                                          "color": {"mode": "thresholds"},
                                          "thresholds": {"mode": "absolute",
                                                         "steps": thresholds or [{"color": "green", "value": None}]}},
                             "overrides": []},
             "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                         "colorMode": color_mode, "graphMode": graph, "textMode": text_mode,
                         "justifyMode": "auto", "orientation": "auto"}}
        if desc:
            p["description"] = desc
        if time_from:  # panel-level relative time, e.g. "now/d" = since local midnight (browser tz)
            p["timeFrom"] = time_from
            p["hideTimeOverride"] = False
        self.pid += 1
        self.panels.append(p)
        return p

    def table(self, title, targets, w=24, h=8, desc=None, organize=None):
        p = {"id": self.pid, "title": title, "type": "table", "datasource": DS,
             "gridPos": self._place(w, h), "targets": self._t(targets, instant=True),
             "fieldConfig": {"defaults": {}, "overrides": []},
             "options": {"showHeader": True},
             "transformations": [{"id": "merge", "options": {}}] + ([organize] if organize else [])}
        if desc:
            p["description"] = desc
        self.pid += 1
        self.panels.append(p)
        return p

    @staticmethod
    def _t(targets, instant=False):
        out = []
        for i, t in enumerate(targets):
            expr, legend = t[0], t[1]
            tt = {"refId": chr(65 + i), "expr": expr, "legendFormat": legend, "datasource": DS}
            if instant:
                tt.update(instant=True, range=False, format="table")
            out.append(tt)
        return out

    def dash(self, uid, title, tags, refresh="10s", frm="now-3h", links=None):
        return enforce({"uid": uid, "title": title, "tags": tags, "timezone": "browser",
                        "refresh": refresh, "time": {"from": frm, "to": "now"},
                        "schemaVersion": 39, "version": 1, "editable": True,
                        "links": links or [], "panels": self.panels, "templating": {"list": []}})


def overview():
    b = Board()
    b.row("LLM speed — every model / GPU (rig_llm_exporter :9097)")
    for e, gpu in LANES:
        b.stat(f"Decode t/s · {gpu}",
               [(f'rig_llm_decode_tokens_per_second{{gpu_index="{e}"}}', "{{lane}}")], unit="none", w=4,
               desc="Live decode speed while generating (mean of non-zero 1 s samples, 15 s). 0 = idle.")
    for e, gpu in LANES:
        b.stat(f"Prefill t/s · {gpu}",
               [(f'rig_llm_prefill_tokens_per_second{{gpu_index="{e}"}}', "{{lane}}")], unit="none", w=4,
               desc="Live prefill (uncached prompt tokens) speed. 0 = no prefill in the last 15 s.")
    b.ts("Decode t/s by lane (live)", [("rig_llm_decode_tokens_per_second", LLM)], w=12, minv=0,
         desc="Instantaneous decode speed while generating. Gaps to 0 = idle.")
    b.ts("Prefill t/s by lane (live)", [("rig_llm_prefill_tokens_per_second", LLM)], w=12, minv=0,
         desc="Uncached prompt processing speed per lane.")
    b.ts("Effective decode t/s per request (5m)",
         [("rate(rig_llm_generation_tokens_total[5m]) / rate(rig_llm_decode_seconds_total[5m]) > 0", LLM)],
         w=8, minv=0, desc="generated tokens / decode seconds over 5 min = average per-stream speed.")
    b.ts("Generated tokens/s throughput (all streams, 1m)",
         [("rate(rig_llm_generation_tokens_total[1m])", LLM), ("sum(rate(rig_llm_generation_tokens_total[1m]))", "total")],
         w=8, minv=0, desc="Aggregate output tokens per second across all parallel streams.")
    b.ts("Decode t/s per watt (GPU efficiency)",
         [("(rig_llm_decode_tokens_per_second > 0) / on(uuid) group_left() nvidia_smi_power_draw_watts", LLM)],
         w=8, minv=0, decimals=2, desc="Live decode t/s divided by that card's power draw.")

    b.row("LLM engines — load, KV, cache")
    b.ts("Requests running / waiting", [("rig_llm_requests_running", "{{lane}} running"),
                                        ("rig_llm_requests_waiting", "{{lane}} waiting")], w=8, minv=0, decimals=0)
    b.ts("KV cache / context fill", [("rig_llm_kv_usage_ratio", LLM)], unit="percentunit", w=8, minv=0, maxv=1,
         desc="Largest slot context / max context (FN 5090: 256K, FN 3090 lanes: 128K).")
    b.ts("Prefix cache hit ratio (30m)",
         [("rate(rig_llm_prompt_cached_tokens_total[30m]) / (rate(rig_llm_prompt_tokens_total[30m]) > 0)", LLM)],
         unit="percentunit", w=8, minv=0, maxv=1,
         desc="Share of prompt tokens served from cache (token-weighted). Gaps = no requests in the window.")
    b.ts("TTFT-side: prefill seconds per request (15m)",
         [("rate(rig_llm_prefill_seconds_total[15m]) / (rate(rig_llm_requests_total[15m]) > 0)", LLM)],
         unit="s", w=8, minv=0)
    b.ts("Prompt tokens/s (incl. cached) vs computed",
         [("rate(rig_llm_prompt_tokens_total[5m])", "{{lane}} total"),
          ("rate(rig_llm_prompt_tokens_total[5m]) - rate(rig_llm_prompt_cached_tokens_total[5m])", "{{lane}} computed")],
         w=8, minv=0)
    b.ts("Requests/min completed", [("rate(rig_llm_requests_total[5m]) * 60", LLM)], w=8, minv=0)
    b.ts("MTP drafts accepted / offered (15m)", [
        ("rate(rig_llm_drafts_accepted_total[15m]) / (rate(rig_llm_drafts_offered_total[15m]) > 0)", LLM)],
        unit="percentunit", w=8, minv=0, maxv=1, desc="Strata MTP speculative decoding acceptance per lane.")
    b.ts("Host RAM available (shared expert arena lives in RAM)",
         [('node_memory_MemAvailable_bytes{job="node"}', "MemAvailable"), ('node_memory_SwapFree_bytes{job="node"}', "Swap free")],
         unit="bytes", w=8, minv=0, desc="Three Strata lanes share one 50 GB expert arena (/mnt/strata-arena); "
                                          "the engines keep >= 2.5 GiB free for parking conversations.")
    b.ts("Engine up", [("rig_llm_up", LLM)], w=8, minv=0, maxv=1, decimals=0, draw="line")

    b.row("GPUs")
    b.ts("GPU util %", [("nvidia_smi_utilization_gpu_ratio" + GPU_JOIN, "{{gpu}}")], unit="percentunit", w=8, minv=0, maxv=1)
    b.ts("GPU power W", [("nvidia_smi_power_draw_watts" + GPU_JOIN, "{{gpu}}")], unit="watt", w=8, minv=0)
    b.ts("GPU core temp C", [("nvidia_smi_temperature_gpu" + GPU_JOIN, "{{gpu}}")], unit="celsius", w=8)
    b.ts("GPU VRAM / hotspot C (gputemps)",
         [('rig_gpu_sensor_temp_celsius{sensor=~"vram|junction"}' + GPU_JOIN, "{{gpu}} {{sensor}}")], unit="celsius", w=8)
    b.ts("VRAM used", [("nvidia_smi_memory_used_bytes" + GPU_JOIN, "{{gpu}}")], unit="bytes", w=8, minv=0)
    b.ts("SM clock MHz", [("nvidia_smi_clocks_current_sm_clock_hz / 1e6" + GPU_JOIN, "{{gpu}}")], w=8, minv=0)

    b.row("Power & electricity — measured from Matter wall plugs")
    plug = ("Wall power is metered by GRILLPLATS Matter plugs: \"Rig\" (node 23) and \"Minisforum\" (node 25), "
            "ElectricalPowerMeasurement cluster, meter class ±1%. Rules: "
            "rig/config/prometheus-rules/power.yml, group rig_power_plug. Money is reported in EUR with "
            "a fixed 117.581053 RSD/EUR (open.er-api.com, 2026-11-06). The old estimate (GPU NVML + CPU "
            "RAPL + ~55 W rest, /0.89 PSU) left the dashboard on 06.11 — it under-read by ~2x; its rules "
            "stay as rig:power_system_wall_est_watts for pre-06.11 history.")
    kwh = "sum_over_time(rig:power_system_wall_watts[$__range]) * 15 / 3600 / 1000"
    eur = "sum_over_time(rig:electricity_cost_eur_per_hour[$__range]) * 15 / 3600"
    b.stat("System power now (W, wall plugs)", [("rig:power_system_wall_watts", "Rig + Minisforum sockets")],
           unit="watt", w=4, decimals=0, desc=plug,
           thresholds=[{"color": "green", "value": None}, {"color": "yellow", "value": 400},
                       {"color": "red", "value": 650}])
    b.stat("Cost now (EUR/h)", [("rig:electricity_cost_eur_per_hour", "from wall plugs"),
                                ("rig:electricity_price_eur_per_kwh", "tariff EUR/kWh")],
           unit="currencyEUR", w=4, decimals=3,
           desc="EPS commercial two-rate tariff incl. network, surcharges, excise 7.5%, VAT 20%: "
                "VT 0.2254 / NT 0.1607 EUR/kWh (18.9 / 26.5 RSD, NT 00-08 Belgrade). Fixed monthly fees "
                "excluded. docs/20 (sep-2026 invoice).")
    b.stat("All wall plugs (W)", [("rig:power_all_plugs_watts", "all sockets")], unit="watt", w=4,
           decimals=0, desc="Sum over every device that meters energy (Rig, Minisforum, \"Уголь\").")
    b.stat("Wall plugs reporting", [("count(rig:power_plug_online == 1)", "nodes reporting"),
                                    ("count(rig:power_plug_online)", "total")], w=4, decimals=0,
           graph="none", desc="Metering plugs currently online; a plug that drops out stops its series "
                              "instead of repeating a stale value (mqtt2prometheus cache.timeout 0).")
    b.stat("Today (since 00:00)", [(kwh, "kWh"), (eur, "EUR")], unit="currencyEUR", w=4, decimals=3,
           time_from="now/d", graph="none")
    b.stat("This month", [(kwh, "kWh"), (eur, "EUR")], unit="currencyEUR", w=4, decimals=2,
           time_from="now/M", graph="none")
    b.stat("Dashboard range", [(kwh, "kWh"), (eur, "EUR")], unit="currencyEUR", w=4, decimals=3,
           graph="none", desc="Energy and cost over the time range picked at the top right.")
    b.stat("Month projection (7d avg x 730 h)",
           [("avg_over_time(rig:electricity_cost_eur_per_hour[7d]) * 730", "EUR / month"),
            ("avg_over_time(rig:power_system_wall_watts[7d]) * 730 / 1000", "kWh / month")],
           unit="currencyEUR", w=4, decimals=2, graph="none")
    b.ts("Wall plug power vs GPU/CPU draw", [
        ("rig:power_system_wall_watts", "Rig + Minisforum sockets (measured)"),
        ("rig:power_plug_minisforum_watts", "Minisforum socket"),
        ("rig:power_gpu_watts", "GPU (NVML, per card)"),
        ("rig:power_cpu_watts", "CPU package (RAPL)"),
        ("rig:power_other_watts", "rest: VRM + box (estimate)")], unit="watt", w=16, decimals=0,
        desc="What the socket draws versus the sum of GPU (NVML) and CPU (RAPL). The gap is monitor, "
             "peripherals, disks and PSU losses.")
    b.ts("Cost, EUR/h", [("rig:electricity_cost_eur_per_hour", "from wall plugs")],
         unit="currencyEUR", w=8, decimals=3, minv=0,
         desc="Measured wall power x current tariff (NT 00-08 is ~1/3 cheaper).")
    b.ts("Energy, kWh (integral of measured plug power)", [
        ("(sum_over_time(rig:electricity_kwh_per_hour_plug[$__interval]) * 15 / 3600)",
         "Rig + Minisforum sockets"),
        ("(sum_over_time(rig:electricity_kwh_per_hour_minisforum[$__interval]) * 15 / 3600)",
         "Minisforum socket")], unit="kWh", w=8, decimals=3,
        desc="The rules publish kW/h (15 s interval); sum of steps x 15/3600 = kWh over the selected "
             "window. Not a counter, so increase() does not apply.")
    b.ts("Wall plugs: W / V / A", [
        ('avg_over_time(matter_active_power_watts{sensor=~"Rig|Minisforum|Уголь"}[2m])', "{{sensor}} W"),
        ('avg_over_time(matter_voltage_volts{sensor=~"Rig|Minisforum|Уголь"}[2m])', "{{sensor}} V"),
        ('avg_over_time(matter_current_amps{sensor=~"Rig|Minisforum|Уголь"}[2m])', "{{sensor}} A")],
        w=16, decimals=1,
        desc="Per-socket active power, voltage and current (ElectricalPowerMeasurement). Restricted to "
             "the metering nodes: the exporter's <Cluster>/<Attribute> topic match also catches other "
             "clusters, so an unfiltered selector mixes in unrelated numbers.")

    b.row("Fans")
    b.ts("GPU fan %", [("nvidia_smi_fan_speed_ratio" + GPU_JOIN, "{{gpu}}")], unit="percentunit", w=8, minv=0, maxv=1,
         desc="NVML fan duty per card (3090s: Vix override curve; 5090: VBIOS).")
    b.ts("Board fans RPM", [('node_hwmon_fan_rpm{job="node"} * on(chip,sensor) group_left(fan) rig_fan_info', "{{fan}}")],
         unit="rotrpm", w=8, minv=0,
         desc="nct6799 headers with a tach: radiator (CPU_FAN1), rear (CPU_FAN2), front 3x140 (CHA_FAN), pump. "
              "Bottom 2x P12 Pro on CHA_FAN4 are invisible to Linux (BIOS curve, no tach).")
    b.ts("Board fan PWM %", [('node_hwmon_pwm{job="node",sensor=~"pwm2|pwm3|pwm4|pwm5"} / 255', "{{sensor}}")],
         unit="percentunit", w=8, minv=0, maxv=1,
         overrides=[{"matcher": {"id": "byName", "options": n}, "properties": [{"id": "displayName", "value": v}]}
                    for n, v in (("pwm2", "AIO radiator fans PWM"), ("pwm3", "Rear exhaust PWM"),
                                 ("pwm5", "Front intake PWM"), ("pwm4", "AIO pump PWM"))],
         desc="fancontrol output (all follow CPU Tctl 50-80 C; front floor 110 > rear 75 = positive pressure). "
              "Pump = BIOS fixed 255.")

    b.row("Host")
    b.ts("CPU busy %", [('1 - avg(rate(node_cpu_seconds_total{mode="idle",job="node"}[1m]))', "CPU")],
         unit="percentunit", w=8, minv=0, maxv=1)
    b.ts("CPU temp C", [('node_hwmon_temp_celsius{job="node",chip=~".*18_3",sensor="temp1"}', "CPU package (Tctl)")],
         unit="celsius", w=8)
    b.ts("RAM available", [('node_memory_MemAvailable_bytes{job="node"}', "available"),
                           ('node_memory_SwapFree_bytes{job="node"}', "swap free")], unit="bytes", w=8, minv=0)
    b.ts("Load / disk free", [('node_load1{job="node"}', "load1")], w=8, minv=0,
         overrides=[{"matcher": {"id": "byName", "options": "/ free"},
                     "properties": [{"id": "unit", "value": "bytes"}, {"id": "custom.axisPlacement", "value": "right"}]}])["targets"].append(
        {"refId": "B", "expr": 'node_filesystem_avail_bytes{mountpoint="/",job="node"}', "legendFormat": "/ free", "datasource": DS})

    b.row("LiteLLM gateway (summary — full view: dashboard 'LiteLLM gateway')")
    b.ts("LiteLLM req/s by model", [("sum by (requested_model) (rate(litellm_proxy_total_requests_metric_total[5m]))", "{{requested_model}}")], w=8, minv=0)
    b.ts("LiteLLM failed req/s", [("sum by (requested_model, exception_status) (rate(litellm_proxy_failed_requests_metric_total[5m]))", "{{requested_model}} {{exception_status}}")], w=8, minv=0)
    b.ts("LiteLLM in-flight", [("sum(litellm_in_flight_requests)", "in flight")], w=8, minv=0, decimals=0)
    return b.dash("rig-overview", "Rig overview", ["rig"],
                  links=[{"title": "LiteLLM gateway", "type": "link", "url": "/d/rig-litellm", "icon": "dashboard"},
                         {"title": "Node full", "type": "dashboards", "tags": ["linux"]}])


def litellm():
    b = Board()
    q = "requested_model"
    b.row("Traffic")
    b.stat("Req/s (5m)", [("sum(rate(litellm_proxy_total_requests_metric_total[5m]))", "req/s")], w=4, decimals=2)
    b.stat("Failed req/s (5m)", [("sum(rate(litellm_proxy_failed_requests_metric_total[5m])) or vector(0)", "failed/s")], w=4,
           decimals=3, thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 0.05}, {"color": "red", "value": 0.5}])
    b.stat("Success % (1h)", [("sum(increase(litellm_proxy_total_requests_metric_total{status_code=~\"2..\"}[1h])) / "
                               "sum(increase(litellm_proxy_total_requests_metric_total[1h]))", "success")],
           unit="percentunit", w=4, thresholds=[{"color": "red", "value": None}, {"color": "orange", "value": 0.95}, {"color": "green", "value": 0.99}])
    b.stat("In-flight now", [("sum(litellm_in_flight_requests)", "in flight")], w=4, decimals=0)
    b.stat("Output tok/s (5m)", [("sum(rate(litellm_output_tokens_metric_total[5m]))", "out tok/s")], w=4)
    b.stat("Requests (24h)", [("sum(increase(litellm_proxy_total_requests_metric_total[24h]))", "24h")], w=4, decimals=0)
    b.ts("Requests/s by model", [(f"sum by ({q}) (rate(litellm_proxy_total_requests_metric_total[5m]))", f"{{{{{q}}}}}")], w=8, minv=0)
    b.ts("Requests/s by status code", [("sum by (status_code) (rate(litellm_proxy_total_requests_metric_total[5m]))", "{{status_code}}")], w=8, minv=0)
    b.ts("Requests/s by client key", [("sum by (api_key_alias, client_ip) (rate(litellm_proxy_total_requests_metric_total[5m]))", "{{api_key_alias}} {{client_ip}}")], w=8, minv=0,
         desc="api_key_alias 'None' = master key (ops/admin or a misconfigured client).")
    b.ts("Failures/s by model · exception", [(f"sum by ({q}, exception_class, exception_status, route) (rate(litellm_proxy_failed_requests_metric_total[5m]))",
                                              f"{{{{{q}}}}} {{{{route}}}} {{{{exception_status}}}} {{{{exception_class}}}}")], w=12, minv=0)
    b.ts("Deployment failures/s (upstream engines)", [("sum by (model_group, api_base, exception_status) (rate(litellm_deployment_failure_responses_total[5m]))",
                                                        "{{model_group}} {{api_base}} {{exception_status}}")], w=12, minv=0)

    b.row("Latency")
    for title, m in (("Total request latency", "litellm_request_total_latency_metric"),
                     ("Time to first token", "litellm_llm_api_time_to_first_token_metric"),
                     ("Upstream LLM API latency", "litellm_llm_api_latency_metric")):
        b.ts(f"{title} p50 / p95 by model",
             [(f"histogram_quantile(0.5, sum by (le, {q}) (rate({m}_bucket[15m])))", f"p50 {{{{{q}}}}}"),
              (f"histogram_quantile(0.95, sum by (le, {q}) (rate({m}_bucket[15m])))", f"p95 {{{{{q}}}}}")],
             unit="s", w=8, minv=0)
    b.ts("Gateway overhead (avg)", [("sum by (model_group) (rate(litellm_overhead_latency_metric_sum[15m])) / "
                                     "sum by (model_group) (rate(litellm_overhead_latency_metric_count[15m]))", "{{model_group}}")],
         unit="s", w=8, minv=0, desc="Time LiteLLM itself adds per request.")
    b.ts("Queue time (avg)", [(f"sum by ({q}) (rate(litellm_request_queue_time_seconds_sum[15m])) / "
                               f"sum by ({q}) (rate(litellm_request_queue_time_seconds_count[15m]))", f"{{{{{q}}}}}")],
         unit="s", w=8, minv=0)
    b.ts("Latency per output token (avg)", [("sum by (model_group) (rate(litellm_deployment_latency_per_output_token_sum[15m])) / "
                                             "sum by (model_group) (rate(litellm_deployment_latency_per_output_token_count[15m]))", "{{model_group}}")],
         unit="s", w=8, minv=0)

    b.row("Tokens")
    b.ts("Input tokens/s by model", [(f"sum by ({q}) (rate(litellm_input_tokens_metric_total[5m]))", f"{{{{{q}}}}}")], w=8, minv=0)
    b.ts("Output tokens/s by model", [(f"sum by ({q}) (rate(litellm_output_tokens_metric_total[5m]))", f"{{{{{q}}}}}"),
                                      (f"sum by ({q}) (rate(litellm_output_reasoning_tokens_metric_total[5m]))", f"{{{{{q}}}}} reasoning")], w=8, minv=0)
    b.ts("Cached input share", [(f"sum by ({q}) (rate(litellm_input_cached_tokens_metric_total[30m])) / "
                                 f"(sum by ({q}) (rate(litellm_input_tokens_metric_total[30m])) > 0)", f"{{{{{q}}}}}")],
         unit="percentunit", w=8, minv=0, maxv=1, desc="As reported by the engine in usage.prompt_tokens_details.")
    b.ts("Tokens/s by client key", [("sum by (api_key_alias) (rate(litellm_total_tokens_metric_total[5m]))", "{{api_key_alias}}")], w=12, minv=0)
    b.ts("Tokens (24h, by model)", [(f"sum by ({q}) (increase(litellm_total_tokens_metric_total[24h]))", f"{{{{{q}}}}}")], w=12, minv=0)

    b.row("Keys, budgets, limits")
    b.ts("Spend $/h by key (fake pricing → budgets)", [("sum by (api_key_alias) (rate(litellm_spend_metric_total[1h])) * 3600", "{{api_key_alias}}")],
         unit="currencyUSD", w=8, minv=0, desc="Synthetic $0.05/$0.15 per Mtok model_info pricing (runaway guard), not real money.")
    b.ts("Remaining budget by key", [("max by (api_key_alias) (litellm_remaining_api_key_budget_metric{api_key_alias!=\"\"} < 1e12)", "{{api_key_alias}}"),
                                     ("max by (api_key_alias) (litellm_api_key_max_budget_metric)", "{{api_key_alias}} max")],
         unit="currencyUSD", w=8, minv=0)
    b.ts("RPM used / allowed by key", [("max by (api_key_alias) (litellm_api_key_rate_limit_used_metric{rate_limit_type=\"requests\"})", "{{api_key_alias}} used"),
                                        ("max by (api_key_alias) (litellm_api_key_rate_limit_allowed_metric{rate_limit_type=\"requests\"})", "{{api_key_alias}} limit")],
         w=8, minv=0)
    b.ts("Deployment state (0 healthy, 1 partial, 2 outage)", [("max by (litellm_model_name, api_base) (litellm_deployment_state)", "{{litellm_model_name}} {{api_base}}")],
         w=12, minv=0, maxv=2, decimals=0)
    b.ts("Budget reset in (hours)", [("max by (api_key_alias) (litellm_api_key_budget_remaining_hours_metric)", "{{api_key_alias}}")],
         unit="h", w=12, minv=0)
    return b.dash("rig-litellm", "LiteLLM gateway", ["rig", "litellm"], refresh="30s", frm="now-6h",
                  links=[{"title": "Rig overview", "type": "link", "url": "/d/rig-overview", "icon": "dashboard"}])


def main(argv):
    if len(argv) > 1 and argv[1] == "--enforce":
        for f in argv[2:]:
            d = json.load(open(f))
            json.dump(enforce(d), open(f, "w"), indent=1)
            print("enforced", f)
        return
    for name, d in (("rig-overview.json", overview()), ("rig-litellm.json", litellm())):
        (OUT / name).write_text(json.dumps(d, indent=1) + "\n")
        print("wrote", OUT / name, len(d["panels"]), "panels")


if __name__ == "__main__":
    main(sys.argv)
