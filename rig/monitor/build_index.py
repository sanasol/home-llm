#!/usr/bin/env python3
"""Build rig/monitor pages as a faithful port of the Strata design.

Reads the Strata design kit from strata-ref/ (pulled from http://rig:8001/web/),
inlines tokens.css + components.css + app.css verbatim, the sprite, the Outfit
fonts (base64) and uPlot (vendored in assets/, canvas charts) into two
single-file pages:

  index.html  - the monitor: the eight Strata metric cards, the state cards,
                the context gauge, the requests table, the conversation cache.
                Adapted to rig: two engines, three GPUs. Every card chart is a
                real uPlot with time axes, a per-series live legend and a
                cursor tooltip listing every series at the hovered instant.
  hw.html     - the hardware page: every hwmon fan/temperature in the same
                design language, one uPlot per sensor, tooltips on hover.

Both share the same header + Monitor/Hardware navigation and the rig.theme
dark/light choice; canvas colors re-resolve from the theme CSS vars on toggle.

Run from rig/monitor/: python3 build_index.py
"""
import base64
import pathlib
import re

HERE = pathlib.Path(__file__).parent
REF = HERE / "strata-ref"

def datauri(path):
    b64 = base64.b64encode((HERE / "assets" / path).read_bytes()).decode()
    return "url(\"data:font/woff2;base64," + b64 + "\")"

tokens = (REF / "tokens.css").read_text()
tokens = tokens.replace('url("../fonts/outfit-latin-wght.woff2")', datauri("outfit.woff2"))
tokens = tokens.replace('url("../fonts/outfit-latin-ext-wght.woff2")', datauri("outfit-ext.woff2"))
CSS = tokens + "\n" + (REF / "components.css").read_text() + "\n" + (REF / "app.css").read_text()
CHART_JS = (HERE / "assets" / "chart.min.js").read_text()

SPRITE = (REF / "sprite.svg").read_text().strip()
SPRITE = re.sub(r"^<svg[^>]*>", '<svg style="display:none" aria-hidden="true">', SPRITE, count=1)

def header(active):
    mon = ' aria-selected="true"' if active == "monitor" else ""
    hw = ' aria-selected="true"' if active == "hw" else ""
    return f"""<header class="st-header">
  <span class="st-header__brand">Rig</span>
  <nav class="st-tabs" role="tablist" aria-label="Views">
    <a class="st-tab" role="tab" href="/" aria-label="Monitor"{mon}><svg class="st-icon st-icon--sm"><use href="#i-activity"/></svg>Monitor</a>
    <a class="st-tab" role="tab" href="/hw" aria-label="Hardware"{hw}><svg class="st-icon st-icon--sm"><use href="#i-cpu"/></svg>Hardware</a>
  </nav>
  <span class="st-header__spacer"></span>
  <span class="updated-at">updated <span id="updated">–</span></span>
  <span class="st-pill" id="pill" data-state="idle" role="status" aria-live="polite">
    <span class="st-pill__dot"></span><span id="pill-text">Connecting&hellip;</span></span>
  <button class="st-btn st-btn--icon" id="theme-btn" title="Light / dark" aria-label="Switch between light and dark">
    <svg class="st-icon"><use id="theme-icon" href="#i-moon"/></svg></button>
</header>"""

PREPAINT = """<script>
  // the theme before the first paint: the saved choice, else the system's
  try {
    const t = localStorage.getItem("rig.theme") ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.dataset.theme = t;
  } catch (e) {}
</script>"""

THEME_JS = """
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("rig.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `#i-${t === "dark" ? "sun" : "moon"}`);
  if (window.onTheme) window.onTheme();
}
$("theme-btn").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let saved = null;
  try { saved = localStorage.getItem("rig.theme"); } catch (err) { /* ignore */ }
  if (!saved) setTheme(e.matches ? "dark" : "light", false);
});
setTheme(document.documentElement.dataset.theme || "light", false);
"""

HELPERS = """
const $ = (id) => document.getElementById(id);
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // bytes -> binary GB
const since = (t) => { if (!t) return ""; const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${fmt(s / 60)} min ago` : `${fmt(s / 3600, 1)} h ago`; };
function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v]) =>
    `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
}
// canvas colors cannot use CSS vars; resolve them per draw (theme change re-renders)
const cvar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const SC = () => [cvar("--st-accent"), cvar("--st-info"), cvar("--st-warn"), cvar("--st-ink-muted")];
const hhmmss = (ms) => new Date(ms).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
const hhmm = (ms) => new Date(ms).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});

// Chart.js line chart: time on the x axis, hover tooltip listing every series at that
// instant, legend below for multi-series. defs: [{label, c: () => color, values, unit, dec, dash, w}]
function mkU(el, xs, defs, opt = {}) {
  el.innerHTML = "";
  const cv = document.createElement("canvas");
  el.appendChild(cv);
  const mk = () => new Chart(cv, {
    type: "line",
    data: {labels: xs, datasets: defs.map((d) => ({
        label: d.label, data: d.values, borderColor: d.c(), borderWidth: d.w || 1.8,
        yAxisID: d.axis || "y",
      borderDash: d.dash ? [4, 3] : [], pointRadius: 0, pointHoverRadius: 3.5,
      tension: 0.2, spanGaps: false, fill: false,
    }))},
    options: {
      responsive: true, maintainAspectRatio: false, animation: false, resizeDelay: 250,
      interaction: {mode: "index", intersect: false},
      plugins: {
        legend: opt.legend === false ? {display: false}
          : {position: "bottom",
            labels: {boxWidth: 8, boxHeight: 8, padding: 8, font: {size: 10}, color: cvar("--st-ink-muted")}},
        tooltip: {
          displayColors: true, usePointStyle: true, boxWidth: 8, boxHeight: 8, padding: 8, cornerRadius: 6,
          backgroundColor: cvar("--st-surface"), titleColor: cvar("--st-ink"), bodyColor: cvar("--st-ink-soft"),
          borderColor: cvar("--st-line"), borderWidth: 1, titleFont: {size: 11}, bodyFont: {size: 11},
          callbacks: {
            title: (items) => hhmmss(items[0].chart.data.labels[items[0].dataIndex]),
            label: (it) => {
              const d = defs[it.datasetIndex];
              return ` ${it.dataset.label}: ${fmt(it.parsed.y, d.dec == null ? 1 : d.dec)}${d.unit || ""}`;
            },
          },
        },
      },
      scales: {
        x: {display: opt.axes !== false, grid: {color: cvar("--st-line")}, border: {color: cvar("--st-line")},
            ticks: {color: cvar("--st-ink-muted"), font: {size: 10}, maxTicksLimit: 4, autoSkip: true,
                    callback(i) {
                      const v = this.getLabelForValue(i);
                      if (v == null) return "";
                      const ls = this.chart.data.labels;
                      const span = ls.length > 1 ? (ls[ls.length - 1] - ls[0]) / 1000 : 0;
                      return span > 1200 ? hhmm(v) : hhmmss(v);
                    }}},
        y: {display: opt.axes !== false, min: opt.autoMin ? undefined : (opt.min == null ? 0 : opt.min), max: opt.max || undefined,
            grace: opt.grace,
            grid: {color: cvar("--st-line")}, border: {display: false},
            ticks: {color: cvar("--st-ink-muted"), font: {size: 10}, maxTicksLimit: 4,
                    callback(v) { return fmt(v, 0) + (opt.yu || ""); }}},
        y1: {display: false, max: opt.preMax || undefined, grace: "8%", grid: {drawOnChartArea: false}},
      },
    },
  });
  return mk();
}
function updU(c, xs, defs) {
  c.data.labels = xs;
  defs.forEach((d, i) => {
    const ds = c.data.datasets[i];
    ds.data = d.values; ds.label = d.label; ds.borderColor = d.c(); ds.borderDash = d.dash ? [4, 3] : [];
  });
  const o = c.options;
  o.plugins.tooltip.backgroundColor = cvar("--st-surface");
  o.plugins.tooltip.titleColor = cvar("--st-ink");
  o.plugins.tooltip.bodyColor = cvar("--st-ink-soft");
  o.plugins.tooltip.borderColor = cvar("--st-line");
  if (o.plugins.legend.labels) o.plugins.legend.labels.color = cvar("--st-ink-muted");
  o.scales.x.grid.color = cvar("--st-line"); o.scales.x.border.color = cvar("--st-line");
  o.scales.y.grid.color = cvar("--st-line");
  o.scales.x.ticks.color = cvar("--st-ink-muted"); o.scales.y.ticks.color = cvar("--st-ink-muted");
  c.update("none");
}
"""

# ================================================================ monitor page
MON_JS = HELPERS + THEME_JS + r"""
// ---------------------------------------------------------------- the metric cards (Strata's own layout)
const METRICS = [
  {key: "speed", label: "Speed", icon: "gauge", unit: "t/s", dual: true},
  {key: "gpu", label: "GPU load", icon: "gpu", unit: "%"},
  {key: "vram", label: "VRAM", icon: "layers", unit: "GB"},
  {key: "temp", label: "GPU temp", icon: "thermometer", unit: "°C"},
  {key: "power", label: "Power", icon: "bolt", unit: "W"},
  {key: "pcie", label: "PCIe", icon: "link", unit: "", noChart: true},
  {key: "cpu", label: "CPU", icon: "cpu", unit: "%"},
  {key: "disk", label: "Disk I/O", icon: "disk", unit: "MB/s"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    ${m.dual ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">–</span><span class="st-metric__sub">decode, both engines</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">–</span><span class="st-metric__sub">prefill, both engines</span></div>
    </div>` : `<span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>`}
    ${m.noChart ? "" : `<div class="mchart" id="mc-${m.key}"></div>`}
  </div></div>`).join("");

function setMetric(key, value, unit, sub) {
  const el = $(`mv-${key}`);
  if (el) el.innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  const s = $(`ms-${key}`);
  if (s) s.textContent = sub || "";
}

// ---------------------------------------------------------------- the Chart.js charts in the cards
const MCH = {};   // key -> {c, defs}
function metricChart(key, xs, defs, opt) {
  const el = $("mc-" + key);
  if (!el) return;
  if (!MCH[key]) {
    MCH[key] = {defs, c: mkU(el, xs, defs, opt)};
  } else {
    const o = MCH[key].c.options;
    if (o.scales.y) o.scales.y.max = opt.max || undefined;
    if (o.scales.y1) o.scales.y1.max = opt.preMax || undefined;
    MCH[key].defs = defs;
    updU(MCH[key].c, xs, defs);
  }
}
window.onTheme = () => { if (lastState) render(lastState); };

// ---------------------------------------------------------------- conversation cache (Strata's own card)
function renderConvCache(c) {
  $("cc-card").hidden = !c;
  if (!c) return;
  const pct = (a, b) => (b ? `${Math.min(100, (100 * a) / b)}%` : "0%");
  $("cc-bars").hidden = !c.enabled;
  if (c.enabled) {
    $("cc-slots-text").textContent = `${fmt(c.parked)} / ${fmt(c.slots)}`;
    $("cc-slots-bar").style.width = pct(c.parked, c.slots);
    const budget = c.budget_mib * 1048576;
    $("cc-mem-text").textContent = `${gb(c.bytes)} / ${gb(budget)} GB`;
    $("cc-mem-bar").style.width = pct(c.bytes, budget);
  }
  $("cc-sum").textContent = c.requests ? `${fmt(c.requests_reused)} of ${fmt(c.requests)} requests reused part of their prompt` : "";
  const share = c.prompt_tokens ? ` (${fmt((100 * c.reused_tokens) / c.prompt_tokens)}% of all prompt tokens)` : "";
  const event = c.last_event ? `${c.last_event === "parked" ? "Parked" : "Restored"} ${fmt(c.last_tokens)} tokens, ${since(c.last_at)}` : null;
  facts($("cc-facts"), [
    ["Last request", c.last_prompt != null ? `${fmt(c.last_reused || 0)} of ${fmt(c.last_prompt)} prompt tokens reused` : null],
    ["Reused since start", c.requests ? `${fmt(c.reused_tokens)} tokens${share}` : null],
    ["Parked / restored", c.enabled ? `${fmt(c.parks)} / ${fmt(c.restores)}${c.evictions ? ` · ${fmt(c.evictions)} evicted` : ""}` : null],
    ["Last switch", c.enabled ? event : null],
  ]);
  $("cc-note").textContent = c.enabled
    ? "A request that continues a parked conversation gets its state back instead of reading it again; the oldest goes when the slots or the memory are full."
    : "The engine keeps the last conversation's state, so a follow-up reads only what is new.";
}

function renderTotals(t, sd) {
  let out = "";
  if (t && t.requests) {
    const since = new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
    const read = t.prompt_tokens - t.reused;
    const pSpeed = t.prompt_ms > 0 && read > 0 ? ` at ${fmt(read / (t.prompt_ms / 1000))} tok/s` : "";
    const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
    out = `Main since ${since}: ${fmt(t.requests)} requests · ${fmt(read)} prompt tokens read${pSpeed} (${fmt(t.reused)} reused) · ${fmt(t.output_tokens)} written${oSpeed}`;
  }
  if (sd && sd.requests) {
    out += (out ? "  —  " : "") + `Side since start: ${fmt(sd.requests)} requests · avg ${fmt(sd.avg_prompt)} in / ${fmt(sd.avg_out)} out`;
  }
  return out;
}

let lastState = null, failures = 0, reqShowAll = false;
function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
}

function render(s) {
  const main = s.main || {}, live = main.live || {}, eng = main.engine || {}, sd = s.side_stats || {};
  const H = s.history || [];
  const VH = H.slice(-300);   // last ~5 minutes in the cards: 900 points on a 240px chart is noise
  const fill = (arr) => { let last = null; return arr.map((v) => (v == null ? last : (last = v))); };  // GPU arrays arrive every 2nd tick
  const hcol = (k) => VH.map((p) => p[k]);
  const gpus = s.gpus || [], host = s.host || {};
  const reqs = s.requests || [], sideReqs = s.side_requests || [];
  const lastMain = reqs[0];

  // header pill: main state, side counts alongside
  let pillText = null;
  if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    pillText = pct != null ? `Reading prompt · ${pct}%` : "Reading prompt";
  } else if (live.state === "generating") {
    pillText = `Generating · ${fmt(live.tok_s, 1)} tok/s`;
  }
  const sideRun = sd.running || 0, sideWait = sd.waiting || 0;
  if (sideRun > 0 || sideWait > 0) {
    const sideTxt = `side ${fmt(sideRun)} run${sideWait ? ` / ${fmt(sideWait)} wait` : ""}`;
    pillText = `${pillText || (live.state === "idle" ? "Idle" : live.state)} · ${sideTxt}`;
  }
  if (live.queued > 0) setPill("queued", `${live.queued} queued · ${pillText || ""}`);
  else if (pillText && live.state !== "idle") setPill(live.state, pillText);
  else setPill("idle", pillText || "Idle");

  // main engine state card (Strata's own logic)
  const on = live.queued > 0 ? "queued" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) b.classList.toggle("on", b.dataset.s === on);
  const prog = $("state-progress");
  let label = "Waiting for a request", detail = "", pct = 0;
  if (live.state === "reading") {
    label = "Reading prompt";
    prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} tokens · ${fmt(pct)}%`;
    } else detail = `${fmt(live.prompt_tokens)} tokens`;
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : "Generating";
    delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = `${fmt(live.generated)} tokens · ${fmt(live.tok_s, 1)} tok/s`;
  } else if (lastMain) {
    delete prog.dataset.tone;
    detail = `last: ${fmt(lastMain.output)} tokens${lastMain.decode_tps ? ` at ${fmt(lastMain.decode_tps, 1)} tok/s` : ""}`;
  }
  $("state-label").textContent = label;
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;

  // side engine state card (same card style, vLLM fields)
  const sOn = sideWait > 0 ? "queued" : sideRun > 0 ? "generating" : "idle";
  for (const b of document.querySelectorAll("#side-badges .st-badge")) b.classList.toggle("on", b.dataset.s === sOn);
  const sProg = $("side-state-progress");
  $("side-state-label").textContent = sideRun || sideWait
    ? `${fmt(sideRun)} running${sideWait ? ` · ${fmt(sideWait)} waiting` : ""}` : "Waiting for a request";
  if (sideRun > 0) sProg.dataset.tone = ""; else delete sProg.dataset.tone;
  $("side-state-bar").style.width = `${Math.min(100, sd.kv_pct || 0)}%`;
  $("side-state-detail").textContent = sd.decode_tps
    ? `KV ${fmt(sd.kv_pct)}% full · ${fmt(sd.decode_tps, 1)} tok/s now`
    : `KV pool ${fmt(sd.kv_tokens || 0)} tokens · ctx ${ctxfmt(sd.max_len || eng.max_context || 0)}`;

  // hardware: one series per GPU everywhere; the card value is the aggregate, the chart legend + tooltip list each card/engine
  const util = gpus.map((g) => g["utilization.gpu"]).filter((v) => v != null);
  const temps = gpus.map((g) => g["temperature.gpu"]).filter((v) => v != null);
  const vramU = gpus.reduce((a, g) => a + (g["memory.used"] || 0), 0) * 1048576;      // MiB -> bytes
  const vramT = gpus.reduce((a, g) => a + (g["memory.total"] || 0), 0) * 1048576;
  const pow = gpus.reduce((a, g) => a + (g["power.draw"] || 0), 0);
  const powL = gpus.reduce((a, g) => a + (g["power.limit"] || 0), 0);
  const lastH = H[H.length - 1] || {};

  const raw = gpus.map((g) => (g.label || g["name"] || `GPU ${g.index}`).replace(/^(NVIDIA |GeForce )/i, "").trim());
  const dup = {};
  raw.forEach((l) => dup[l] = (dup[l] || 0) + 1);
  const seq = {};
  const gLabel = raw.map((l) => dup[l] > 1 ? `${l} #${seq[l] = (seq[l] || 0) + 1}` : l);
  const hcolBy = (k) => {
    const m = Math.max(0, ...VH.map((p) => (p[k] || []).length));
    return Array.from({length: m}, (_, j) => fill(VH.map((p) => (p[k] && p[k][j] != null) ? p[k][j] : null)));
  };
  const lastv = (k) => { for (let i = H.length - 1; i >= 0; i--) { const v = H[i][k]; if (v != null) return v; } return null; };
  const xs = VH.map((p) => p.t * 1000);

  const decMain = live.state === "generating" ? live.tok_s : lastv("main_tps");
  const decSide = sd.decode_tps != null ? sd.decode_tps : lastv("side_tps");
  const tot = (decMain || 0) + (decSide || 0);
  const pfMain = live.state !== "idle" && live.prefill_tok_s_mean ? live.prefill_tok_s_mean : lastv("main_prefill_tps");
  const pfSide = sd.prefill_tps != null ? sd.prefill_tps : lastv("side_prefill_tps");
  const totPf = (pfMain || 0) + (pfSide || 0);
  setMetric("speed", tot ? fmt(tot, 1) : null, "t/s", "");
  const pfEl = $("mv-prefill");
  if (pfEl) pfEl.innerHTML = totPf ? `${esc(fmt(totPf, 0))}<small>t/s</small>` : "–";
  // p99 caps: a bogus post-restart spike would flatten everything; decode and prefill differ 10x, so prefill gets its own axis
  const p99 = (vals, floor) => { const a = vals.filter((x) => x != null && x > 0).sort((x, y) => x - y);
    return a.length ? Math.max(floor, a[Math.min(a.length - 1, Math.floor(a.length * 0.99))] * 1.1) : floor; };
  const decMax = p99([...hcol("main_tps"), ...hcol("side_tps")], 60);
  const preMax = p99([...hcol("main_prefill_tps"), ...hcol("side_prefill_tps")], 200);
  metricChart("speed", xs, [
    {label: "main decode", c: () => SC()[0], values: hcol("main_tps"), unit: " t/s", dec: 1},
    {label: "side decode", c: () => SC()[1], values: hcol("side_tps"), unit: " t/s", dec: 1},
    {label: "main prefill", c: () => SC()[0], dash: true, w: 1.2, values: hcol("main_prefill_tps"), unit: " t/s", dec: 0, axis: "y1"},
    {label: "side prefill", c: () => SC()[1], dash: true, w: 1.2, values: hcol("side_prefill_tps"), unit: " t/s", dec: 0, axis: "y1"},
  ], {max: decMax, preMax});

  setMetric("gpu", util.length ? fmt(util.reduce((a, b) => a + b, 0) / util.length) : null, "%", "average of all GPUs");
  metricChart("gpu", xs, gLabel.map((l, i) =>
    ({label: l, c: () => SC()[i % 3], values: hcolBy("gpu_util_by")[i], unit: "%", dec: 0})), {max: 100, grace: "6%", yu: "%"});

  setMetric("vram", vramT ? gb(vramU) : null, vramT ? `/ ${gb(vramT, 0)} GB` : "GB", "all GPUs");
  {
    const vTot = gpus.map((g) => (g["memory.total"] || 0) / 1024);
    const vby = hcolBy("vram_by").map((arr) => arr.map((x) => x == null ? null : x / 1024));
    metricChart("vram", xs, gLabel.map((l, i) =>
      ({label: l, c: () => SC()[i % 3], values: vby[i], unit: ` GB of ${fmt(vTot[i])} GB`, dec: 1})), {max: Math.max(1, ...vTot), grace: "6%", yu: "G"});
  }

  setMetric("temp", temps.length ? fmt(Math.max(...temps)) : null, "°C", "hottest GPU");
  metricChart("temp", xs, gLabel.map((l, i) =>
    ({label: l, c: () => SC()[i % 3], values: hcolBy("gpu_temp_by")[i], unit: " °C", dec: 0})), {autoMin: true, max: 95, yu: "°"});

  setMetric("power", pow ? fmt(pow) : null, "W", powL ? `of ${fmt(powL)} W total limit` : "");
  metricChart("power", xs, gLabel.map((l, i) =>
    ({label: l, c: () => SC()[i % 3], values: hcolBy("gpu_power_by")[i], unit: ` W (limit ${fmt((gpus[i] || {})["power.limit"] || 0)} W)`, dec: 0})),
    {max: Math.max(1, ...gpus.map((g) => g["power.limit"] || 0)), yu: "W"});

  const g5090 = gpus.find((g) => (g["name"] || "").includes("5090")) || gpus[0] || {};
  setMetric("pcie", g5090["pcie.link.gen.current"] ? `Gen${g5090["pcie.link.gen.current"]}` : null,
            g5090["pcie.link.width.current"] ? `×${g5090["pcie.link.width.current"]}` : "", "main GPU link");

  setMetric("cpu", lastH.cpu_pct == null ? null : fmt(lastH.cpu_pct), "%", host.cpu_name || "");
  metricChart("cpu", xs, [{label: "CPU load", c: () => SC()[0], values: hcol("cpu_pct"), unit: "%", dec: 1}],
              {max: 100, yu: "%", legend: false});

  const dr = lastH.disk_r_mbps;
  setMetric("disk", dr == null ? null : (dr >= 1000 ? fmt(dr / 1024, 2) : fmt(dr, dr < 10 ? 1 : 0)),
            dr >= 1000 ? "GB/s" : "MB/s", "read");
  metricChart("disk", xs, [
    {label: "read", c: () => SC()[1], values: hcol("disk_r_mbps"), unit: " MB/s", dec: 1},
    {label: "write", c: () => SC()[2], values: hcol("disk_w_mbps"), unit: " MB/s", dec: 1},
  ], {max: Math.max(1, ...hcol("disk_r_mbps").filter((x) => x != null), ...hcol("disk_w_mbps").filter((x) => x != null)), yu: "M", legend: false});

  // context fill gauge: main; side KV as a bar row
  const ctx = eng.max_context || 0;
  let used = 0;
  if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (lastMain) used = (lastMain.prompt || 0) + (lastMain.output || 0);
  const frac = ctx ? Math.min(1, used / ctx) : 0;
  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * frac >= 3 ? "1" : "0";
  $("ctx-pct").textContent = `${Math.round(frac * 100)}%`;
  $("ctx-sub").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)}` : "–";
  const cacheBytes = (eng.expert_cache_mib || 0) * 1048576;
  $("slots-text").textContent = eng.expert_slots ? `${fmt(eng.expert_slots)} · ${gb(cacheBytes)} GB` : "–";
  $("slots-bar").style.width = vramT ? `${Math.min(100, (100 * cacheBytes) / vramT)}%` : "0%";
  $("skv-text").textContent = `${fmt(sd.kv_pct || 0)}% · ${fmt(sd.kv_tokens || 0)} tok pool`;
  $("skv-bar").style.width = `${Math.min(100, sd.kv_pct || 0)}%`;
  if ((sd.kv_pct || 0) > 92) $("skv-progress").dataset.tone = "danger"; else delete $("skv-progress").dataset.tone;
  $("ram-text").textContent = host.mem_total ? `${gb(host.mem_total - host.mem_available)} / ${gb(host.mem_total, 0)} GB` : "–";
  const ramPct = host.mem_total ? (100 * (host.mem_total - host.mem_available)) / host.mem_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  const tmax = temps.length ? Math.max(...temps) : null;
  $("temp-text").textContent = tmax == null ? "–" : `${fmt(tmax)} °C`;
  $("temp-bar").style.width = tmax == null ? "0%" : `${Math.min(100, tmax)}%`;

  // requests: both engines, newest first
  const body = $("req-body");
  const rows = [
    ...reqs.map((r) => ({t: r.ts, eng: "main", badge: r.status, prompt: r.prompt, reused: r.reused,
                         out: r.output, pre: r.prefill_tps, dec: r.decode_tps, hit: r.hit, dur: r.dur})),
    ...sideReqs.map((r) => ({t: r.ts, eng: "side", badge: r.status === 200 ? "stop" : String(r.status),
                             prompt: r.prompt, reused: null, out: r.output, pre: r.prefill_tps, dec: r.decode_tps,
                             hit: null, dur: r.dur, approx: r.approx, n: r.n, client: r.client})),
  ].sort((a, b) => (b.t || 0) - (a.t || 0));
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="9" class="muted">No requests yet</td></tr>`;
  } else {
    const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                   disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"]};
    const shown = rows.slice(0, reqShowAll ? rows.length : 12);
    body.innerHTML = shown.map((r) => {
      const [cls, text] = badge[r.badge] || [r.badge && r.badge !== "stop" && /^[0-9]+$/.test(r.badge) ? "st-badge--error" : "", r.badge || "–"];
      const t = new Date(r.t * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const approx = (r.approx && r.out != null) ? "≈" : "";
      const apTitle = r.approx ? ` title="average over ${r.n} completions that finished in one sample tick"` : "";
      const hit = r.hit == null ? "–" : `${(r.hit * 100).toFixed(1)}%`;
      return `<tr><td>${esc(t)}<br><span class="req-src">${r.eng}${r.client ? ` · ${esc(r.client.split(":")[0])}` : ""}</span></td>
        <td><span class="st-badge ${cls}">${esc(text)}</span></td><td class="num">${fmt(r.prompt)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num"${apTitle}>${approx}${fmt(r.out)}</td><td class="num">${fmt(r.pre)}</td><td class="num">${fmt(r.dec, 1)}</td>
        <td class="num">${hit}</td><td class="num">${fmt(r.dur, 1)} s</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  all.hidden = rows.length <= 12;
  all.textContent = reqShowAll ? "Show fewer" : `Show all (${rows.length})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("req-totals").textContent = renderTotals(main.totals, sd);
  $("updated").textContent = new Date((s.updated || Date.now() / 1000) * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});

  facts($("side-facts"), [
    ["TTFT mean", sd.ttft ? `${fmt(sd.ttft, 2)} s` : null],
    ["Speculative decoding", sd.mtp_len ? `draft ${fmt(sd.mtp_len, 1)} tokens · accepted ${fmt(sd.mtp_pct)}%` : null],
    ["Prefix reuse", sd.prefix_pct != null ? `${fmt(sd.prefix_pct)}% of prompt tokens` : null],
    ["KV cache pool", sd.kv_tokens ? `${fmt(sd.kv_tokens)} tokens (${ctxfmt(sd.max_len)} per request)` : null],
    ["Preemptions", sd.preempts != null ? fmt(sd.preempts) : null],
  ]);

  renderConvCache(main.conversation_cache);
}

async function poll() {
  try {
    const r = await fetch("api/state");
    if (r.ok) { lastState = await r.json(); failures = 0; render(lastState); }
    else throw new Error(`HTTP ${r.status}`);
  } catch (e) {
    if (++failures === 3) setPill("error", "Monitor not reachable");
  }
  setTimeout(poll, 1000);
}
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastState) render(lastState); });
poll();
"""

MON_BODY = """
  <div class="monitor">
    <div class="monitor-row">
      <div class="st-card state-card">
        <div class="state-card__head">
          <span class="card-title">Main · Strata :8001 · <span class="muted" id="model-main">qwen3.8-flash-next</span></span>
          <span class="badges" id="state-badges">
            <span class="st-badge" data-s="idle">Idle</span>
            <span class="st-badge st-badge--reading" data-s="reading">Reading</span>
            <span class="st-badge st-badge--generating" data-s="generating">Generating</span>
            <span class="st-badge st-badge--queued" data-s="queued">Queued</span>
          </span>
        </div>
        <div class="state-card__row"><span id="state-label">Waiting for a request</span><span class="muted" id="state-detail"></span></div>
        <div class="st-progress" id="state-progress"><div class="st-progress__bar" id="state-bar" style="width:0%"></div></div>
      </div>
      <div class="st-card state-card">
        <div class="state-card__head">
          <span class="card-title">Side · vLLM :8002 · <span class="muted">qwen3.8-27b</span></span>
          <span class="badges" id="side-badges">
            <span class="st-badge" data-s="idle">Idle</span>
            <span class="st-badge st-badge--generating" data-s="generating">Generating</span>
            <span class="st-badge st-badge--queued" data-s="queued">Queued</span>
          </span>
        </div>
        <div class="state-card__row"><span id="side-state-label">Waiting for a request</span><span class="muted" id="side-state-detail"></span></div>
        <div class="st-progress" id="side-state-progress"><div class="st-progress__bar" id="side-state-bar" style="width:0%"></div></div>
      </div>
    </div>

    <div class="metrics" id="metrics"><!-- filled by the script: the eight Strata cards --></div>

    <div class="monitor-row">
      <div class="st-card ctx-card">
        <span class="card-title">Context fill</span>
        <div class="ctx-gauge">
          <svg class="st-gauge" viewBox="0 0 120 120" aria-hidden="true">
            <circle class="st-gauge__track" cx="60" cy="60" r="50" fill="none" stroke-width="10" stroke-linecap="round"
                    stroke-dasharray="235.6 314.2" transform="rotate(135 60 60)"/>
            <circle class="st-gauge__fill" id="ctx-fill" cx="60" cy="60" r="50" fill="none" stroke-width="10"
                    stroke-linecap="round" stroke-dasharray="0 314.2" transform="rotate(135 60 60)"/>
          </svg>
          <div class="ctx-gauge__label"><span class="ctx-gauge__pct" id="ctx-pct">0%</span><span class="muted" id="ctx-sub">&ndash;</span></div>
        </div>
        <div class="bar-row"><span>Experts in VRAM</span><span class="muted" id="slots-text">&ndash;</span></div>
        <div class="st-progress"><div class="st-progress__bar" id="slots-bar" style="width:0%"></div></div>
        <div class="bar-row"><span>Side KV fill</span><span class="muted" id="skv-text">&ndash;</span></div>
        <div class="st-progress" id="skv-progress"><div class="st-progress__bar" id="skv-bar" style="width:0%"></div></div>
        <div class="bar-row"><span>System RAM</span><span class="muted" id="ram-text">&ndash;</span></div>
        <div class="st-progress" id="ram-progress"><div class="st-progress__bar" id="ram-bar" style="width:0%"></div></div>
        <div class="bar-row"><span>GPU temperature (max)</span><span class="muted" id="temp-text">&ndash;</span></div>
        <div class="st-progress" data-tone="warn"><div class="st-progress__bar" id="temp-bar" style="width:0%"></div></div>
      </div>
      <div class="st-card req-card">
        <div class="req-head"><span class="card-title">Recent requests</span>
          <button class="st-btn st-btn--secondary" id="req-all" hidden>Show all</button></div>
        <div class="table-wrap" id="req-wrap">
          <table class="st-table">
            <thead><tr><th>Time</th><th>Status</th><th class="num">Prompt</th><th class="num">Reused</th><th class="num">Output</th>
              <th class="num" title="New prompt tokens per second, excluding reused cache tokens (engine prompt time)">Prefill t/s</th><th class="num">Decode t/s</th>
              <th class="num" title="Main only: the share of the experts looked up that were already in VRAM">VRAM hit rate</th><th class="num">Duration</th></tr></thead>
            <tbody id="req-body"><tr><td colspan="9" class="muted">No requests yet</td></tr></tbody>
          </table>
        </div>
        <p class="muted req-totals" id="req-totals"></p>
      </div>
    </div>

    <div class="monitor-row">
      <div class="st-card cc-card" id="cc-card">
        <div class="req-head"><span class="card-title">Conversation cache</span><span class="muted small" id="cc-sum"></span></div>
        <div class="cc-bars" id="cc-bars" hidden>
          <div><div class="bar-row"><span>Parked conversations</span><span class="muted" id="cc-slots-text">&ndash;</span></div>
            <div class="st-progress"><div class="st-progress__bar" id="cc-slots-bar" style="width:0%"></div></div></div>
          <div><div class="bar-row"><span>Their memory (RAM)</span><span class="muted" id="cc-mem-text">&ndash;</span></div>
            <div class="st-progress"><div class="st-progress__bar" id="cc-mem-bar" style="width:0%"></div></div></div>
        </div>
        <dl class="facts" id="cc-facts"></dl>
        <p class="muted small cc-note" id="cc-note"></p>
      </div>
      <div class="st-card">
        <div class="req-head"><span class="card-title">Side engine · vLLM details</span></div>
        <dl class="facts" id="side-facts"></dl>
      </div>
    </div>
  </div>
"""

# ================================================================ hardware page
HW_JS = HELPERS + THEME_JS + r"""
// history buffers (client-side, ~2 min per sensor); one uPlot per sensor, updated every second
const buf = new Map(), CAP = 120;
function push(k, v) {
  if (v == null || isNaN(v)) return;
  let a = buf.get(k);
  if (!a) { a = {x: [], y: []}; buf.set(k, a); }
  a.x.push(Date.now()); a.y.push(Number(v));
  if (a.y.length > CAP) { a.x.shift(); a.y.shift(); }
}
const TONE = {ok: () => SC()[0], warn: () => SC()[1], bad: () => SC()[2]};
const lvl = (v, w, c) => v >= c ? "bad" : v >= w ? "warn" : "ok";
function thrFor(chip, label) {
  if (chip === "k10temp") return [70, 85];
  if (chip === "nvme") return [50, 70];
  if (chip.startsWith("spd")) return [60, 80];
  if (chip === "nct6799") return (label === "TSI0_TEMP" || label === "SMBUSMASTER 0") ? [80, 100] : [65, 85];
  return [75, 100];
}
const CHIP = {k10temp: "CPU", nvme: "NVMe SSD", nct6799: "Motherboard · NCT6799", spd5118: "Memory (SPD)",
              prom21_xhci: "USB 3.2 controller", "r8169_0_900:00": "Ethernet · Realtek", mt7925_phy0: "Wi-Fi",
              amdgpu: "Integrated GPU"};
function chipName(c) { for (const k in CHIP) if (c === k || c.startsWith(k)) return CHIP[k]; return c; }
const SENSOR = {"Tctl": "CPU package", "Tccd1": "CPU core complex 1", "Tccd2": "CPU core complex 2",
  "Composite": "Drive composite", "Sensor 1": "Flash chips", "Sensor 2": "Controller",
  "SYSTIN": "System (board)", "CPUTIN": "CPU socket probe", "TSI0_TEMP": "VRM zone (TSI0)", "SMBUSMASTER 0": "SMBus controller",
  "AUXTIN0": "Board aux 1", "AUXTIN1": "Board aux 2", "AUXTIN4": "Board aux 3", "PCH_CHIP_TEMP": "Chipset (PCH)",
  "PCH_CHIP_CPU_MAX_TEMP": "PCH–CPU max", "edge": "GPU edge"};
function sensorName(chip, label, id) {
  if (SENSOR[label]) return SENSOR[label];
  if (chip.startsWith("r8169")) return "Network PHY";
  if (chip.startsWith("mt7925")) return "Wi-Fi chip";
  if (chip === "prom21_xhci") return "Controller";
  if (chip.startsWith("spd")) return "DIMM slot";
  return label || ("Sensor " + id);
}
const HC = {};   // key -> Chart
const unitOf = (k) => k.startsWith("rpm") ? " RPM" : k.startsWith("blower") ? " %" : k === "cpuGHz" ? " GHz" : " °C";
const decOf = (k) => k.startsWith("rpm") || k.startsWith("blower") ? 0 : 1;
function chartFor(key, color, name, h) {
  const b = buf.get(key), el = document.querySelector(`[data-hc="${key}"]`);
  if (!el) return;
  const defs = () => [{label: name, c: color, unit: unitOf(key), dec: decOf(key), values: b ? b.y.slice() : []}];
  if (!HC[key]) {
    HC[key] = mkU(el, b ? b.x.slice() : [], defs(), {axes: false, legend: false,
      autoMin: !key.startsWith("rpm") && !key.startsWith("blower"),
      max: key.startsWith("rpm") ? 3300 : null});
  } else {
    updU(HC[key], b.x.slice(), defs());
  }
}
window.onTheme = () => { if (lastHw) render(lastHw); };

let lastHw = null, hwSig = "";
function render(s) {
  lastHw = s;
  $("updated").textContent = new Date((s.updated || Date.now() / 1000) * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
  const hw = s.hw || [], host = s.host || {}, gpus = s.gpus || [];
  const tempsOf = (name) => { for (const c of hw) if (c.chip === name) return c.temps.filter((t) => t.c >= 5); return []; };

  const kt = tempsOf("k10temp");
  const tctl = (kt.find((t) => t.label === "Tctl") || {}).c;
  const tccd1 = (kt.find((t) => t.label === "Tccd1") || {}).c;
  const tccd2 = (kt.find((t) => t.label === "Tccd2") || {}).c;
  const nv = tempsOf("nvme");
  const nvWorst = nv.length ? Math.max(...nv.map((t) => t.c)) : null;
  const dimm = hw.filter((c) => c.chip.startsWith("spd")).flatMap((c) => c.temps).filter((t) => t.c >= 5);
  const dimmWorst = dimm.length ? Math.max(...dimm.map((t) => t.c)) : null;

  // feed the history buffers
  push("cpuTctl", tctl); push("cpuGHz", host.cpu_ghz);
  for (const g of gpus) push("gput" + g.index, g["temperature.gpu"]);
  push("nvmeW", nvWorst); push("dimmW", dimmWorst); push("cpuTdie", host.cpu_temp);
  for (const c of hw) for (const f of (c.fans || [])) push("rpm" + c.chip + f.id, f.rpm);
  gpus.forEach((g) => push("blower" + g.index, g["fan.speed"] ?? 0));
  for (const c of hw) for (const t of c.temps) if (t.c >= 5) push("sen" + c.chip + t.id, t.c);

  // rebuild the DOM only when the sensor set changes; between rebuilds update values + charts
  const sig = ["cpuTctl", "cpuGHz", ...gpus.map((g) => "gput" + g.index), nvWorst != null ? "nvmeW" : "",
    dimmWorst != null ? "dimmW" : "", host.cpu_temp != null ? "cpuTdie" : "",
    ...hw.flatMap((c) => (c.fans || []).map((f) => "rpm" + c.chip + f.id)),
    ...gpus.map((g) => "blower" + g.index),
    ...hw.flatMap((c) => c.temps.filter((t) => t.c >= 5).map((t) => "sen" + c.chip + t.id))].join(",");

  if (sig !== hwSig) {
    hwSig = sig;
    for (const k in HC) { try { HC[k].destroy(); } catch (e) { /* ignore */ } delete HC[k]; }
    const card = (key, name, ic, val, sub, tone, h) => `
      <div class="st-card"><div class="st-metric">
        <span class="st-metric__label">${icon(ic, "st-icon st-icon--sm")}${esc(name)}</span>
        <span class="st-metric__value" id="v-${key}" ${tone ? `style="color:${TONE[tone]()}"` : ""}>${val}</span>
        <span class="st-metric__sub">${esc(sub || "")}</span>
        <div class="hchart" data-hc="${key}"></div>
      </div></div>`;
    const cards = [];
    if (tctl != null) {
      cards.push(card("cpuTctl", "CPU package", "thermometer", `${fmt(tctl)}<small>°C</small>`,
        `${fmt(host.cpu_ghz, 2)} GHz · max safe 95°`, lvl(tctl, 70, 85)));
    }
    cards.push(card("cpuGHz", "CPU clock", "bolt", `${fmt(host.cpu_ghz, 2)}<small>GHz</small>`,
      `load ${fmt(host.load1, 1)} · ${fmt(host.load5, 1)}`, null));
    for (const g of gpus) {
      cards.push(card("gput" + g.index, g.label || "GPU", "gpu", `${fmt(g["temperature.gpu"])}<small>°C</small>`,
        `fan ${fmt(g["fan.speed"])}% · ${fmt(g["power.draw"], 0)} W`, lvl(g["temperature.gpu"] ?? 0, 65, 85)));
    }
    if (nvWorst != null) cards.push(card("nvmeW", "NVMe SSD", "disk", `${fmt(nvWorst)}<small>°C</small>`, "throttles around 70°", lvl(nvWorst, 50, 70)));
    if (dimmWorst != null) cards.push(card("dimmW", "Memory (SPD)", "memory", `${fmt(dimmWorst)}<small>°C</small>`, `${dimm.length} DIMM sensors`, lvl(dimmWorst, 60, 80)));
    if (host.cpu_temp != null) cards.push(card("cpuTdie", "CPU (Tdie)", "cpu", `${fmt(host.cpu_temp)}<small>°C</small>`, `CCDs ${fmt(tccd1)}° / ${fmt(tccd2)}°`, lvl(host.cpu_temp, 70, 85)));
    $("temps-strip").innerHTML = cards.join("");

    const fanCards = [];
    for (const c of hw) for (const f of (c.fans || [])) {
      const key = "rpm" + c.chip + f.id, on = f.rpm > 0;
      fanCards.push(card(key, `${chipName(c.chip)} · header ${f.id}`, "activity",
        on ? `${fmt(f.rpm)}<small>RPM</small>` : "–", on ? "" : "nothing connected", on ? "warn" : null, 36));
    }
    gpus.forEach((g) => fanCards.push(card("blower" + g.index, `${g.label || "GPU"} · blower`, "activity",
      `${fmt(g["fan.speed"] ?? 0)}<small>%</small>`, "", null, 36)));
    $("fans-grid").innerHTML = fanCards.join("");

    const groups = {};
    for (const c of hw) {
      const ts = c.temps.filter((t) => t.c >= 5);
      if (!ts.length) continue;
      groups[c.chip] = groups[c.chip] || [];
      groups[c.chip].push({chip: c.chip, seq: groups[c.chip].length, ts});
    }
    const order = ["nct6799", "nvme", "k10temp", "spd5118", "r8169_0_900:00", "mt7925_phy0", "prom21_xhci", "amdgpu"];
    const keys = Object.keys(groups).sort((a, b) => (order.indexOf(a) + 99) - (order.indexOf(b) + 99));
    const chips = [];
    for (const chip of keys) for (const g of groups[chip]) {
      const suffix = groups[chip].length > 1 ? ` #${g.seq + 1}` : "";
      const rows = g.ts.map((t) => {
        const key = "sen" + chip + t.id, nm = sensorName(chip, t.label, t.id);
        return `<div class="hw-sensor"><div class="bar-row"><span>${esc(nm)}</span>
          <span class="muted" id="v-${key}">–</span></div>
          <div class="hchart" data-hc="${key}"></div></div>`;
      }).join("");
      chips.push(`<div class="st-card"><span class="card-title">${esc(chipName(chip))}${suffix}</span>${rows}</div>`);
    }
    $("chips-grid").innerHTML = chips.join("");
  }

  // live values + tone colors
  const setV = (key, txt, tone) => {
    const el = $("v-" + key);
    if (!el) return;
    el.innerHTML = txt;
    el.style.color = tone ? TONE[tone]() : "";
  };
  if (tctl != null) setV("cpuTctl", `${fmt(tctl)}<small>°C</small>`, lvl(tctl, 70, 85));
  setV("cpuGHz", `${fmt(host.cpu_ghz, 2)}<small>GHz</small>`, null);
  for (const g of gpus) setV("gput" + g.index, `${fmt(g["temperature.gpu"])}<small>°C</small>`, lvl(g["temperature.gpu"] ?? 0, 65, 85));
  if (nvWorst != null) setV("nvmeW", `${fmt(nvWorst)}<small>°C</small>`, lvl(nvWorst, 50, 70));
  if (dimmWorst != null) setV("dimmW", `${fmt(dimmWorst)}<small>°C</small>`, lvl(dimmWorst, 60, 80));
  if (host.cpu_temp != null) setV("cpuTdie", `${fmt(host.cpu_temp)}<small>°C</small>`, lvl(host.cpu_temp, 70, 85));
  for (const g of gpus) setV("blower" + g.index, `${fmt(g["fan.speed"] ?? 0)}<small>%</small>`, null);
  for (const c of hw) for (const f of (c.fans || [])) setV("rpm" + c.chip + f.id, f.rpm > 0 ? `${fmt(f.rpm)}<small>RPM</small>` : "–", f.rpm > 0 ? "warn" : null);
  for (const c of hw) for (const t of c.temps) if (t.c >= 5) {
    const [w, cr] = thrFor(c.chip, t.label);
    setV("sen" + c.chip + t.id, `${fmt(t.c, 1)} °C`, lvl(t.c, w, cr));
  }

  chartFor("cpuTctl", () => TONE[lvl(tctl ?? 0, 70, 85)](), "CPU package");
  chartFor("cpuGHz", () => SC()[1], "CPU clock");
  for (const g of gpus) chartFor("gput" + g.index, () => TONE[lvl(g["temperature.gpu"] ?? 0, 65, 85)](), g.label || "GPU");
  if (nvWorst != null) chartFor("nvmeW", () => TONE[lvl(nvWorst, 50, 70)](), "NVMe SSD");
  if (dimmWorst != null) chartFor("dimmW", () => TONE[lvl(dimmWorst, 60, 80)](), "Memory (SPD)");
  if (host.cpu_temp != null) chartFor("cpuTdie", () => TONE[lvl(host.cpu_temp, 70, 85)](), "CPU (Tdie)");
  for (const c of hw) for (const f of (c.fans || [])) chartFor("rpm" + c.chip + f.id, () => SC()[1], `${chipName(c.chip)} · header ${f.id}`, 36);
  gpus.forEach((g) => chartFor("blower" + g.index, () => SC()[1], `${g.label || "GPU"} · blower`, 36));
  for (const c of hw) for (const t of c.temps) if (t.c >= 5) {
    const [w, cr] = thrFor(c.chip, t.label);
    chartFor("sen" + c.chip + t.id, () => TONE[lvl(t.c, w, cr)](), sensorName(c.chip, t.label, t.id), 34);
  }

  // pill: the worst sensor overall
  let worst = -1e9, where = "";
  if (tctl != null && tctl > worst) { worst = tctl; where = "CPU"; }
  for (const g of gpus) if ((g["temperature.gpu"] ?? -1) > worst) { worst = g["temperature.gpu"]; where = g.label; }
  for (const t of nv) if (t.c > worst) { worst = t.c; where = "NVMe"; }
  if (dimmWorst != null && dimmWorst > worst) { worst = dimmWorst; where = "DIMM"; }
  const l = worst >= 85 ? "bad" : worst >= 70 ? "warn" : "ok";
  $("pill").dataset.state = l === "bad" ? "alert" : l === "warn" ? "queued" : "idle";
  $("pill-text").textContent = (l === "bad" ? "Hot: " : l === "warn" ? "Warm: " : "All normal · ") + where + " " + fmt(worst, 0) + " °C";
}

async function poll() {
  try {
    const r = await fetch("api/state");
    if (r.ok) render(await r.json());
  } catch (e) {
    $("pill").dataset.state = "alert";
    $("pill-text").textContent = "Monitor not reachable";
  }
  setTimeout(poll, 1000);
}
poll();
"""

HW_BODY = """
  <div class="monitor">
    <section class="st-card hw-section">
      <span class="card-title">Core temperatures</span>
      <p class="muted small">The hottest sensor drives the status pill. CPU package is the one that matters: max safe 95 °C, the engine throttles itself before that.</p>
      <div class="metrics hw-strip" id="temps-strip"></div>
    </section>

    <section class="st-card hw-section">
      <span class="card-title">Fans</span>
      <p class="muted small">Board headers come from the NCT6799 super I/O; a 0 RPM header means nothing is plugged into it. GPU coolers report percent, not RPM.</p>
      <div id="fans-grid" class="hw-grid"></div>
    </section>

    <section class="st-card hw-section">
      <span class="card-title">All sensors by chip</span>
      <p class="muted small">Every chip Linux exposes under <code>/sys/class/hwmon</code>. Dead channels (0 / −62 °C) are hidden. Colors: green normal, amber warm, red hot — thresholds per chip type.</p>
      <div id="chips-grid" class="hw-grid hw-grid--wide"></div>
    </section>
  </div>
"""

# ================================================================ assemble
def page(title, active, body, js, extra_css=""):
    return f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{CSS}
/* rig additions on top of the kit */
.req-src{{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--st-ink-muted)}}
a.st-tab{{text-decoration:none}}
.updated-at{{font-size:var(--st-fs-xs);color:var(--st-ink-muted)}}
.st-pill[data-state="alert"] .st-pill__dot{{background:var(--st-danger);box-shadow:0 0 0 4px var(--st-danger-tint)}}
.hw-section{{width:100%}}
.hw-strip{{grid-template-columns:repeat(auto-fill,minmax(230px,1fr))}}
.hw-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:var(--st-s-3)}}
.hw-grid--wide{{grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:var(--st-s-4)}}
.hw-sensor{{margin-top:10px}}
.speed-values{{display:flex;gap:var(--st-s-5);align-items:baseline}}
.speed-prefill .st-metric__value{{font-size:var(--st-fs-lg)}}
.mchart{{width:100%;height:112px;margin-top:2px}}
.hchart{{width:100%;height:46px;margin-top:2px}}
</style>
{PREPAINT}
</head>
<body>
{SPRITE}
{header(active)}
<main>
  <div class="view view--monitor">
{body}  </div>
</main>
<script>
{CHART_JS}
</script>
<script>
{js}
</script>
</body>
</html>
"""

(HERE / "index.html").write_text(page("Rig · monitor", "monitor", MON_BODY, MON_JS))
(HERE / "hw.html").write_text(page("Rig · hardware", "hw", HW_BODY, HW_JS))
print("wrote index.html + hw.html")
