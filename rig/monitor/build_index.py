#!/usr/bin/env python3
"""Build rig/monitor pages as a faithful port of the Strata design.

Reads the Strata design kit from strata-ref/ (pulled from http://rig:8001/web/),
inlines tokens.css + components.css + app.css verbatim, the sprite, the Outfit
fonts (base64) and uPlot (vendored in assets/, canvas charts) into two
single-file pages:

  index.html  - the monitor: the eight Strata metric cards, the state cards,
                the context gauge, the requests table, the conversation cache.
                Adapted to rig: three Strata lanes, three GPUs. Every card chart is a
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
    hist = ' aria-selected="true"' if active == "hist" else ""
    return f"""<header class="st-header">
  <span class="st-header__brand">Rig</span>
  <nav class="st-tabs" role="tablist" aria-label="Views">
    <a class="st-tab" role="tab" href="/" aria-label="Monitor"{mon}><svg class="st-icon st-icon--sm"><use href="#i-activity"/></svg>Monitor</a>
    <a class="st-tab" role="tab" href="/history" aria-label="History"{hist}><svg class="st-icon st-icon--sm"><use href="#i-clock"/></svg>History</a>
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
// chart axes: 24h clock, no AM/PM - short labels never rotate in a narrow card
const hhmmss = (ms) => new Date(ms).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23"});
const hhmm = (ms) => new Date(ms).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", hourCycle: "h23"});

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
            ticks: {color: cvar("--st-ink-muted"), font: {size: 10}, maxTicksLimit: 4, autoSkip: true, maxRotation: 0,
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
  {key: "pcie", label: "PCIe traffic", icon: "link", unit: ""},
  {key: "cpu", label: "CPU", icon: "cpu", unit: "%"},
  {key: "disk", label: "Disk I/O", icon: "disk", unit: "MB/s"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    ${m.dual ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">–</span><span class="st-metric__sub" id="ms-speed">decode</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">–</span><span class="st-metric__sub" id="ms-prefill">prefill</span></div>
    </div>` : m.key === "power" ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-power">–</span><span class="st-metric__sub" id="ms-power">wall · metered socket</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-gpupow">–</span><span class="st-metric__sub" id="ms-gpupow">GPUs</span></div>
    </div>
    <span class="st-metric__sub" id="ms-energy1"></span>
    <span class="st-metric__sub" id="ms-energy2"></span>` : `<span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>`}
    <div class="mchart" id="mc-${m.key}"></div>
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

let lastState = null, failures = 0, reqShowAll = 0, REQF = "all";
document.querySelectorAll("[data-reqf]").forEach((b) => b.addEventListener("click", () => {
  REQF = b.dataset.reqf;
  document.querySelectorAll("[data-reqf]").forEach((x) => x.classList.toggle("on", x === b));
  if (lastState) render(lastState);
}));
let RANGE = "live", histData = null, histGot = 0;     // chart window: live | 1h | 24h | 7d
let SUMR = "24h";                                      // summary window

function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
}

function render(s) {
  const EIDS = ["main", "l2", "l3"];
  const names = s.eng_names || {};
  const engTag = Object.fromEntries(EIDS.map((e) => [e, (names[e] || {}).label || e]));
  window.ENGTAG = engTag;
  for (const el of document.querySelectorAll("[data-lbl]")) el.textContent = engTag[el.dataset.lbl] || el.dataset.lbl;
  const META = Object.fromEntries(EIDS.map((e) => [e, (s.eng[e] || {}).meta || {}]));
  const BOOT = Object.fromEntries(EIDS.map((e) => [e, (s.eng[e] || {}).boot || null]));
  const LV = Object.fromEntries(EIDS.map((e) => [e, ((s.eng[e] || {}).live || {}).live || {}]));
  const ST = Object.fromEntries(EIDS.map((e) => [e, ((s.eng[e] || {}).stats) || {}]));
  const live = LV.main, eng = ((s.eng.main || {}).live || {}).engine || {}, cache = ((s.eng.main || {}).live || {}).conversation_cache;
  const sm = ST.main;
  const H = RANGE === "live" ? (s.history || []) : (histData || []);
  const VH = RANGE === "live" ? H.slice(-300) : H;   // live: last ~5 min; history ranges come pre-downsampled
  const fill = (arr) => { let last = null; return arr.map((v) => (v == null ? last : (last = v))); };
  const hcol = (k) => VH.map((p) => p[k]);
  const gpus = s.gpus || [], host = s.host || {};
  const reqs = (s.requests || []).slice(-200);
  const lastOf = (e) => { let x = null; for (const r of reqs) if (r.eng === e && (!x || r.ts > x.ts)) x = r; return x; };
  const lastMain = lastOf("main");

  // header pill: the busiest lane first, the other busy lanes appended
  const act = EIDS.filter((e) => LV[e].state === "reading" || LV[e].state === "generating");
  const queued = EIDS.reduce((a, e) => a + (LV[e].queued || 0), 0);
  const laneText = (e) => {
    const l = LV[e];
    if (l.state === "reading") {
      const pct = l.prompt_total ? Math.round((100 * l.prompt_read) / l.prompt_total) : null;
      return `${engTag[e]} · reading${pct != null ? ` ${pct}%` : ""}`;
    }
    return `${engTag[e]} · ${fmt(l.tok_s, 1)} tok/s`;
  };
  const bootTxt = EIDS.filter((e) => BOOT[e]).map((e) => `${engTag[e]} · ${BOOT[e].state === "starting" ? `starting ${Math.round(BOOT[e].pct)}%` : BOOT[e].phase}`);
  const pillText = [...act.map(laneText), ...bootTxt].join(" · ");
  if (queued > 0) setPill("queued", `${queued} queued${pillText ? ` · ${pillText}` : ""}`);
  else if (act.length) setPill(LV[act[0]].state, pillText);
  else if (bootTxt.length) setPill("queued", pillText);
  else setPill("idle", "Idle");

  // ---- three engine state cards (all Strata lanes, same logic)
  for (const e of EIDS) {
    const l = LV[e], st = ST[e], lastR = lastOf(e), boot = BOOT[e], meta = META[e];
    $(`${e}-tag`).textContent = engTag[e];
    $(`${e}-sub`).textContent = (names[e] || {}).sub || "";
    $(`${e}-sub`).title = meta.model ? `model ${meta.model} · unit ${meta.unit} · preset ${meta.preset}` : "";
    const on = boot ? (boot.state === "starting" ? "starting" : "down") : l.queued > 0 ? "queued" : (l.state || "idle");
    for (const bd of document.querySelectorAll(`#${e}-badges .st-badge`)) {
      bd.classList.toggle("on", bd.dataset.s === on);
      // while a lane boots or is down only that badge shows (same row height); otherwise the four normal ones
      bd.style.display = bd.classList.contains("opt") ? (bd.dataset.s === on ? "" : "none") : (boot ? "none" : "");
    }
    $(`${e}-card`).classList.toggle("booting", !!boot);
    const note = $(`${e}-note`);
    note.textContent = (meta.pending || []).length ? `preset changed, applies on restart: ${meta.pending.join(", ")}` : "";
    const prog = $(`${e}-progress`);
    let label = "Waiting for a request", detail = "", pct = 0;
    if (boot) {
      const secs = boot.elapsed_s != null && boot.elapsed_s >= 0 ? ` · ${boot.elapsed_s < 120 ? boot.elapsed_s + " s" : Math.round(boot.elapsed_s / 60) + " min"}` : "";
      label = boot.state === "starting" ? `Starting: ${boot.phase}` : boot.state === "failed" ? "Failed" : `Down: ${boot.phase}`;
      detail = boot.state === "starting" ? `${Math.round(boot.pct)}%${secs}` : "";
      if (boot.detail) note.textContent = boot.detail;
      pct = boot.state === "starting" ? boot.pct : 0;
      prog.dataset.tone = boot.state === "starting" ? "warn" : "danger";
    } else if (l.state === "reading") {
      label = "Reading prompt"; prog.dataset.tone = "info";
      if (l.prompt_total) {
        pct = (100 * l.prompt_read) / l.prompt_total;
        detail = `${fmt(l.prompt_read)} / ${fmt(l.prompt_total)} tokens · ${fmt(pct)}%`;
      } else detail = `${fmt(l.prompt_tokens)} tokens`;
    } else if (l.state === "generating") {
      label = l.phase ? l.phase[0].toUpperCase() + l.phase.slice(1) : "Generating";
      delete prog.dataset.tone;
      pct = l.max_tokens ? Math.min(100, (100 * l.generated) / l.max_tokens) : 0;
      detail = `${fmt(l.generated)} tokens · ${fmt(l.tok_s, 1)} tok/s · ctx ${fmt(st.ctx_pct || 0)}%`;
    } else {
      delete prog.dataset.tone;
      if (!Object.keys(st).length) label = "Engine not reachable";
      detail = lastR ? `last: ${fmt(lastR.output)} tokens${lastR.decode_tps ? ` at ${fmt(lastR.decode_tps, 1)} tok/s` : ""}`
                     : `ctx held ${fmt(st.ctx_pct || 0)}% · prefix reuse ${fmt(st.prefix_pct)}%`;
    }
    $(`${e}-label`).textContent = label;
    $(`${e}-detail`).textContent = detail;
    $(`${e}-bar`).style.width = `${pct}%`;
  }
  const TPS = window.TPS || (window.TPS = {});
  const ago = (ts) => { const s0 = Math.max(0, Date.now() / 1000 - ts);
    return s0 < 90 ? `${Math.round(s0)}s` : s0 < 5400 ? `${Math.round(s0 / 60)}m` : `${Math.round(s0 / 3600)}h`; };
  function setTps(id, hkey, v) {
    const b = TPS[id] || (TPS[id] = {q: [], ts: 0, last: null});
    if (v != null && v > 0) { b.q.push(v); if (b.q.length > 5) b.q.shift(); b.ts = Date.now(); b.last = {v, ts: Date.now() / 1000}; }
    if (Date.now() - b.ts > 10000) b.q = [];
    if (!b.last) {
      const isPre = id.endsWith("-pre");
      const lim = isPre ? 20000 : Infinity;  // guard against one counter spike seeding the "last" line
      let idx = -1;
      for (let i = H.length - 1; i >= 0; i--) { const hv = H[i][hkey]; if (hv && hv <= lim) { idx = i; break; } }
      if (idx >= 0) {
        let sum = 0, n = 0;
        for (let i = idx; i >= Math.max(0, idx - 7); i--) {  // average the run of the last busy points, not one sample
          const hv = H[i][hkey];
          if (!hv || hv > lim) break;
          sum += hv; n++;
        }
        b.last = {v: sum / n, ts: H[idx].t};
      }
    }
    $(id).textContent = b.q.length ? fmt(b.q.reduce((a, x) => a + x, 0) / b.q.length, 1) : "–";
    const el = $(id + "-last");
    if (el) el.textContent = (!b.q.length && b.last) ? `last ${fmt(b.last.v, 1)} t/s · ${ago(b.last.ts)} ago` : "";
  }
  for (const e of EIDS) { setTps(`${e}-dec`, `${e}_tps`, ST[e].decode_tps); setTps(`${e}-pre`, `${e}_prefill_tps`, ST[e].prefill_tps); }

  // hardware: one series per GPU everywhere
  const util = gpus.map((g) => g["utilization.gpu"]).filter((v) => v != null);
  const temps = gpus.map((g) => g["temperature.gpu"]).filter((v) => v != null);
  const vramU = gpus.reduce((a, g) => a + (g["memory.used"] || 0), 0) * 1048576;
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

  // speed: three lanes, each with its own line (decode solid, prefill dashed on the hidden y1 axis)
  const dec = {}, pf = {};
  for (const e of EIDS) {
    dec[e] = ST[e].decode_tps != null ? ST[e].decode_tps : null;
    pf[e] = ST[e].prefill_tps != null ? ST[e].prefill_tps : null;
  }
  const tot = EIDS.reduce((a, e) => a + (dec[e] || 0), 0);
  const totPf = EIDS.reduce((a, e) => a + (pf[e] || 0), 0);
  const subOf = (o, d) => EIDS.map((e) => `${engTag[e].replace("FN ", "")} ${fmt(o[e] != null ? o[e] : lastv(e + d), 0)}`).join(" · ");
  setMetric("speed", tot ? fmt(tot, 1) : null, "t/s", subOf(dec, "_tps"));
  const pfEl = $("mv-prefill");
  if (pfEl) pfEl.innerHTML = totPf ? `${esc(fmt(totPf, 0))}<small>t/s</small>` : "–";
  const pfSub = $("ms-prefill");
  if (pfSub) pfSub.textContent = subOf(pf, "_prefill_tps");
  const p99 = (vals, floor) => { const a = vals.filter((x) => x != null && x > 0).sort((x, y) => x - y);
    return a.length ? Math.max(floor, a[Math.min(a.length - 1, Math.floor(a.length * 0.99))] * 1.1) : floor; };
  const decMax = p99(EIDS.flatMap((e) => hcol(e + "_tps")), 60);
  const preMax = p99(EIDS.flatMap((e) => hcol(e + "_prefill_tps")), 200);
  metricChart("speed", xs, [
    ...EIDS.map((e, i) => ({label: engTag[e], c: () => SC()[i], values: hcol(e + "_tps"), unit: " t/s", dec: 1})),
    ...EIDS.map((e, i) => ({label: `${engTag[e]} prefill`, c: () => SC()[i], dash: true, w: 1.2, values: hcol(e + "_prefill_tps"), unit: " t/s", dec: 0, axis: "y1"})),
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

  // power: measured wall W from the Matter plug (rig:power_system_wall_watts) big, GPUs (NVML) small;
  // kWh/RSD lines. The GPU+CPU estimate survives only as the dashed history line (it is what pre-06.11
  // history holds) and as the fallback when the plug stops reporting.
  {
    const en = s.energy || {};
    const ok = en.ok && Date.now() / 1000 - (en.t || 0) < 90;
    const metered = ok ? en.wall_w : null;
    const estNow = ok && en.psu_k && pow ? (pow + (en.cpu_w || 0) + (en.other_w || 0)) * en.psu_k : null;
    const sysNow = metered != null ? metered : estNow;
    setMetric("power", sysNow != null ? fmt(sysNow) : null, "W",
              metered != null ? "wall · metered socket" : (ok ? "wall · estimate (no plug data)" : "system · no data"));
    const gp = $("mv-gpupow");
    if (gp) gp.innerHTML = pow ? `${esc(fmt(pow))}<small>W</small>` : "–";
    const gs = $("ms-gpupow");
    if (gs) gs.textContent = powL ? `GPUs of ${fmt(powL)} W limit` : "GPUs";
    const tariff = en.price == null ? "" : en.price < 20 ? " · night tariff" : " · day tariff";
    $("ms-energy1").textContent = ok
      ? `≈ ${fmt(sysNow != null && en.price != null ? sysNow / 1000 * en.price : en.rsd_h, 1)} RSD/h${tariff} · today ${fmt(en.today_kwh, 1)} kWh ≈ ${fmt(en.today_rsd)} RSD` : "";
    $("ms-energy2").textContent = ok
      ? `month ${fmt(en.month_kwh, 0)} kWh ≈ ${fmt(en.month_rsd)} RSD · forecast ≈ ${fmt(en.month_proj_rsd, 0)} RSD/mo` : "";
    const sysv = hcol("sys_power");
    const sysMax = Math.max(0, ...sysv.filter((x) => x != null));
    metricChart("power", xs, [
      {label: "system (est.)", c: () => SC()[3], dash: true, w: 1.6, values: fill(sysv), unit: " W (GPU+CPU estimate)", dec: 0},
      ...gLabel.map((l, i) =>
        ({label: l, c: () => SC()[i % 3], values: hcolBy("gpu_power_by")[i], unit: ` W (limit ${fmt((gpus[i] || {})["power.limit"] || 0)} W)`, dec: 0}))],
      {max: Math.max(1, sysMax * 1.08, ...gpus.map((g) => g["power.limit"] || 0)), yu: "W"});
  }

  // PCIe: real NVML Rx/Tx throughput per second (dmon), one pair of lines per GPU
  {
    const rx = hcolBy("pcie_rx_by"), tx = hcolBy("pcie_tx_by");
    const flat = [...rx.flat(), ...tx.flat()].filter((x) => x != null);
    const peak = flat.length ? Math.max(...flat) : 0;
    setMetric("pcie", peak >= 1000 ? fmt(peak / 1000, 1) : fmt(peak), peak >= 1000 ? "GB/s peak" : "MB/s peak", "PCIe gen/width: " + gpus.map((g) => `${gLabel[gpus.indexOf(g)]} Gen${g["pcie.link.gen.current"] || "?"}×${g["pcie.link.width.current"] || "?"}`).join(" · "));
    const defs = [];
    gLabel.forEach((l, i) => {
      if ((tx[i] || []).some((v) => v != null) || (rx[i] || []).some((v) => v != null)) {
        defs.push({label: `${l} tx`, c: () => SC()[i % 3], values: tx[i] || [], unit: " MB/s", dec: 0});
        defs.push({label: `${l} rx`, c: () => SC()[i % 3], dash: true, w: 1.2, values: rx[i] || [], unit: " MB/s", dec: 0});
      }
    });
    metricChart("pcie", xs, defs, {max: Math.max(1, peak), yu: "M"});
  }

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

  // context fill gauge: main; KV fill bars for A and B
  const ctx = eng.max_context || sm.ctx_max || 0;
  let used = 0;
  if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (lastMain) used = (lastMain.prompt || 0) + (lastMain.output || 0);
  else used = ctx * (sm.ctx_pct || 0) / 100;
  const frac = ctx ? Math.min(1, used / ctx) : 0;
  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * frac >= 3 ? "1" : "0";
  $("ctx-pct").textContent = `${Math.round(frac * 100)}%`;
  $("ctx-sub").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)}` : "–";
  const cacheBytes = (eng.expert_cache_mib || 0) * 1048576;
  $("slots-text").textContent = eng.expert_slots ? `${fmt(eng.expert_slots)} · ${gb(cacheBytes)} GB` : "–";
  $("slots-bar").style.width = vramT ? `${Math.min(100, (100 * cacheBytes) / vramT)}%` : "0%";
  for (const e of ["l2", "l3"]) {
    const st = ST[e], cp = st.ctx_pct || 0;
    $(`ctx-${e}-text`).textContent = st.ctx_max ? `${fmt(cp)}% of ${ctxfmt(st.ctx_max)} held` : "–";
    $(`ctx-${e}-bar`).style.width = `${Math.min(100, cp)}%`;
    if (cp > 92) $(`ctx-${e}-progress`).dataset.tone = "danger"; else delete $(`ctx-${e}-progress`).dataset.tone;
  }
  $("ram-text").textContent = host.mem_total ? `${gb(host.mem_total - host.mem_available)} / ${gb(host.mem_total, 0)} GB` : "–";
  const ramPct = host.mem_total ? (100 * (host.mem_total - host.mem_available)) / host.mem_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  const tmax = temps.length ? Math.max(...temps) : null;
  $("temp-text").textContent = tmax == null ? "–" : `${fmt(tmax)} °C`;
  $("temp-bar").style.width = tmax == null ? "0%" : `${Math.min(100, tmax)}%`;

  // requests: all three engines, newest first, model chips color-coded, filterable
  const body = $("req-body");
  const mCls = {main: "m-main", l2: "m-l2", l3: "m-l3"};
  let rows = reqs.map((r) => ({t: r.ts, eng: r.eng, badge: r.status,
    prompt: r.prompt, reused: r.reused, out: r.output, pre: r.prefill_tps, dec: r.decode_tps,
    hit: r.hit, dur: r.dur, approx: r.approx, n: r.n, client: r.client,
    sess: r.sess, conv: r.conv, disk: r.disk_ms})).sort((a, b) => (b.t || 0) - (a.t || 0));
  if (REQF !== "all") rows = rows.filter((r) => r.eng === REQF);
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="11" class="muted">No requests ${REQF === "all" ? "" : "for this model"} yet</td></tr>`;
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
      return `<tr class="${mCls[r.eng] || ""}"><td>${esc(t)}<br><span class="req-src ${mCls[r.eng] || ""}">${esc(engTag[r.eng] || r.eng)}${r.client ? ` · ${esc(r.client.split(":")[0])}` : ""}</span></td>
        <td class="sess"${r.sess ? ` title="session ${esc(r.sess)}${r.conv ? ` · conversation ${esc(r.conv)}` : ""}"` : ""}>${r.sess ? `${esc(r.sess.slice(0, 12))}${r.conv ? `<br><span class="muted">c${esc(r.conv.slice(0, 6))}</span>` : ""}` : "–"}</td>
        <td><span class="st-badge ${cls}">${esc(text)}</span></td><td class="num">${fmt(r.prompt)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num"${apTitle}>${approx}${fmt(r.out)}</td><td class="num">${fmt(r.pre)}</td><td class="num">${fmt(r.dec, 1)}</td>
        <td class="num">${hit}</td><td class="num"${r.disk != null ? ` title="the conversation was restored from disk before this request"` : ""}>${r.disk != null ? `${fmt(r.disk, 0)} ms` : "–"}</td><td class="num">${fmt(r.dur, 1)} s</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  all.style.visibility = rows.length <= 12 ? "hidden" : "visible";
  all.textContent = reqShowAll ? "Show fewer" : `Show all (${rows.length})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("updated").textContent = new Date((s.updated || Date.now() / 1000) * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});

  // engine details: one column per engine
  const factCols = EIDS.map((e) => {
    const st = ST[e];
    const m = META[e], b = BOOT[e];
    return [`${engTag[e]} · Strata :${m.port || "?"}`, [
      ["Status", b ? (b.state === "starting" ? `starting · ${b.phase} · ${Math.round(b.pct)}%` : b.phase) : "serving"],
      ["Model", m.model || null],
      ["Engine", m.version ? `Strata ${m.version}` : null],
      ["GPU", m.gpu_name ? `${m.gpu_name.replace(/^NVIDIA (GeForce )?/, "")} (idx ${m.gpu_index})` : null],
      ["Context window", m.ctx ? `${m.ctx_label} (${fmt(m.ctx)})${m.rope ? ` · YaRN ×${m.rope} over ${ctxfmt(m.trained_ctx || 262144)} trained` : " · native"}` : null],
      ["KV cache", m.kv ? `${m.kv} · ${m.slots} slot${m.slots > 1 ? "s" : ""}` : null],
      ["Images", m.images == null ? null : m.images ? "yes" : "no"],
      ["Unit · preset", m.unit ? `${m.unit} · ${m.preset || "?"}` : null],
      ["Pending restart", (m.pending || []).length ? m.pending.join(", ") : null],
      ["Decode mean", st._c && st._c.dec_s > 1 ? `${fmt(st._c.gen / st._c.dec_s, 1)} t/s since start` : null],
      ["Prefill mean", st._c && st._c.pre_s > 1 ? `${fmt((st._c.prompt - st._c.reused) / st._c.pre_s, 0)} t/s since start` : null],
      ["Prefix reuse", st.prefix_pct != null ? `${fmt(st.prefix_pct)}%` : null],
      ["MTP accepted", st.mtp_pct != null ? `${fmt(st.mtp_pct)}% of drafts` : null],
      ["Context held", st.ctx_max ? `${fmt(st.ctx_pct || 0)}% of ${ctxfmt(st.ctx_max)}` : null],
      ["Experts in VRAM", st.expert_slots ? `${fmt(st.expert_slots)} · ${gb((st.expert_cache_mib || 0) * 1048576)} GB` : null],
      ["Requests", st.requests != null ? fmt(st.requests) : null],
      ["Service", (s.services || {})[e] || null],
    ]];
  });
  $("eng-facts").innerHTML = factCols.map(([title, rows]) =>
    `<div><div class="bar-row"><span>${esc(title)}</span></div><dl class="facts">${
      rows.filter((r) => r[1] != null).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")
    }</dl></div>`).join("");

  renderConvCache(cache);
}

// ------------------------------------------------------------- history range + summary
async function loadHist() {
  try {
    const r = await fetch(`api/history?range=${RANGE}`);
    if (r.ok) { histData = (await r.json()).points; histGot = Date.now(); }
  } catch (e) { /* keep the old window */ }
}
document.querySelectorAll("[data-range]").forEach((b) => b.addEventListener("click", async () => {
  RANGE = b.dataset.range;
  document.querySelectorAll("[data-range]").forEach((x) => x.classList.toggle("on", x === b));
  if (RANGE !== "live") { histData = null; await loadHist(); }
  if (lastState) render(lastState);
}));

function fmtSpan(t0, t1) {
  const d = (t) => new Date(t * 1000).toLocaleString([], {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"});
  return `${d(t0)} — ${d(t1)}`;
}
async function loadSummary() {
  let url = `api/summary?range=${SUMR}`;
  if (SUMR === "custom") {
    const f = $("sum-from").value, t = $("sum-to").value;
    if (!f || !t) return;
    url = `api/summary?from=${new Date(f).getTime() / 1000}&to=${new Date(t).getTime() / 1000}`;
  }
  try {
    const s = await (await fetch(url)).json();
    const names = window.ENGTAG || {main: "main", l2: "lane2", l3: "lane3"};
    $("sum-span").textContent = fmtSpan(s.from, s.to);
    $("sum-body").innerHTML = ["main", "l2", "l3"].map((e) => {
      const x = s.engines[e] || {}, d = x.disk || {};
      const disk = d.saves || d.restores
        ? `<span title="restored from disk ${fmt(d.restores)}x, parked ${fmt(d.saves)}x, RAM-guard refusals ${fmt(d.fails)}x, ${fmt(d.restored_tokens)} tokens loaded back, ${gb(d.bytes)} written, avg restore ${fmt(d.avg_restore_ms, 0)} ms">↓${fmt(d.restores)} ↑${fmt(d.saves)}${d.fails ? ` <span class="st-badge st-badge--queued">${fmt(d.fails)}✗</span>` : ""}<br><span class="muted">${d.hit_pct != null ? `${d.hit_pct}% · ${fmt(d.avg_restore_ms, 0)} ms` : ""}</span></span>`
        : "–";
      return `<tr class="m-${e}"><td><span class="req-src m-${e}">${names[e]}</span></td><td class="num">${fmt(x.requests)}</td>
        <td class="num">${fmt(x.computed)}${x.reused ? ` <span class="muted">+${kfmt(x.reused)}</span>` : ""}</td>
        <td class="num">${fmt(x.output)}</td>
        <td class="num">${fmt(x.prefill_tps)}</td><td class="num">${fmt(x.decode_tps, 1)}</td>
        <td class="num">${fmt(x.avg_prompt)} / ${fmt(x.avg_output)}</td>
        <td class="num">${disk}</td></tr>`;
    }).join("");
  } catch (e) { $("sum-body").innerHTML = `<tr><td colspan="8" class="muted">summary unavailable</td></tr>`; }
}
document.querySelectorAll("[data-sum]").forEach((b) => b.addEventListener("click", () => {
  SUMR = b.dataset.sum;
  document.querySelectorAll("[data-sum]").forEach((x) => x.classList.toggle("on", x === b));
  $("sum-custom").hidden = SUMR !== "custom";
  loadSummary();
}));
$("sum-custom").addEventListener("change", () => loadSummary());

async function poll() {
  try {
    const r = await fetch("api/state");
    if (r.ok) { lastState = await r.json(); failures = 0; if (RANGE !== "live" && (!histData || Date.now() - histGot > 60000)) await loadHist(); render(lastState);
      if (!window.SUM_NAMED) { window.SUM_NAMED = 1; loadSummary(); } }
    else throw new Error(`HTTP ${r.status}`);
  } catch (e) {
    if (++failures === 3) setPill("error", "Monitor not reachable");
  }
  setTimeout(poll, 1000);
}
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastState) render(lastState); });
loadSummary();
poll();
"""

MON_BODY = """
  <div class="monitor">
    <div class="monitor-row eng-row">
""" + "".join(f"""      <div class="st-card state-card eng-card m-{e}" id="{e}-card">
        <div class="state-card__head">
          <span class="card-title"><span class="req-src m-{e}" id="{e}-tag">{e}</span></span>
          <span class="badges" id="{e}-badges">
            <span class="st-badge" data-s="idle">Idle</span>
            <span class="st-badge st-badge--reading" data-s="reading">Reading</span>
            <span class="st-badge st-badge--generating" data-s="generating">Generating</span>
            <span class="st-badge st-badge--queued" data-s="queued">Queued</span>
            <span class="st-badge st-badge--queued opt" data-s="starting">Starting</span>
            <span class="st-badge st-badge--error opt" data-s="down">Down</span>
          </span>
        </div>
        <div class="eng-sub muted" id="{e}-sub"></div>
        <div class="state-card__row"><span id="{e}-label">Waiting for a request</span><span class="muted" id="{e}-detail"></span></div>
        <div class="st-progress" id="{e}-progress"><div class="st-progress__bar" id="{e}-bar" style="width:0%"></div></div>
        <div class="eng-note" id="{e}-note"></div>
        <div class="eng-tps">
          <div class="eng-tps__m"><span class="st-metric__value" id="{e}-dec">–</span><small>decode t/s</small><small class="tps-last" id="{e}-dec-last"></small></div>
          <div class="eng-tps__m"><span class="st-metric__value" id="{e}-pre">–</span><small>prefill t/s</small><small class="tps-last" id="{e}-pre-last"></small></div>
        </div>
      </div>
""" for e in ("main", "l2", "l3")) + """    </div>

    <div class="rangebar">
      <span class="muted small">charts:</span>
      <button class="st-btn st-btn--secondary on" data-range="live">Live 5 min</button>
      <button class="st-btn st-btn--secondary" data-range="1h">1 hour</button>
      <button class="st-btn st-btn--secondary" data-range="24h">1 day</button>
      <button class="st-btn st-btn--secondary" data-range="7d">7 days</button>
      <button class="st-btn st-btn--secondary" data-range="30d">30 days</button>
    </div>

    <div class="metrics" id="metrics"><!-- filled by the script: the eight Strata cards --></div>

    <div class="monitor-row">
      <div class="st-card req-card" style="grid-column:1/-1">
        <div class="req-head req-head--f"><span class="card-title">Recent requests</span>
          <span class="rangebar req-filters">
            <button class="st-btn st-btn--secondary on" data-reqf="all">all</button>
            <button class="st-btn st-btn--secondary c-main" data-reqf="main" data-lbl="main">main</button>
            <button class="st-btn st-btn--secondary c-l2" data-reqf="l2" data-lbl="l2">lane2</button>
            <button class="st-btn st-btn--secondary c-l3" data-reqf="l3" data-lbl="l3">lane3</button>
          </span>
          <button class="st-btn st-btn--secondary" id="req-all" style="visibility:hidden">Show all</button></div>
        <div class="table-wrap" id="req-wrap">
          <table class="st-table">
            <thead><tr><th>Time · model</th><th title="LiteLLM session id (first 12 chars) and the conversation key the auto-sessions park under">Session</th><th>Status</th><th class="num">Prompt</th><th class="num">Reused</th><th class="num">Output</th>
              <th class="num" title="New prompt tokens per second, excluding reused cache tokens (engine prompt time)">Prefill t/s</th><th class="num">Decode t/s</th>
              <th class="num" title="The share of the experts looked up that were already in VRAM">VRAM hit rate</th><th class="num" title="How long the conversation was restored from disk (auto-sessions) before this request">Disk</th><th class="num">Duration</th></tr></thead>
            <tbody id="req-body"><tr><td colspan="11" class="muted">No requests yet</td></tr></tbody>
          </table>
        </div>
        <p class="muted req-totals" id="req-totals">Requests are kept 90 days — the History tab searches all of them (the totals card below sums any period).</p>
      </div>
    </div>

    <div class="monitor-row">
      <div class="st-card ctx-card">
        <span class="card-title">Context fill · <span data-lbl="main">main</span></span>
        <div class="ctx-gauge">
          <svg class="st-gauge" viewBox="0 0 120 120" aria-hidden="true">
            <circle class="st-gauge__track" cx="60" cy="60" r="50" fill="none" stroke-width="10" stroke-linecap="round"
                    stroke-dasharray="235.6 314.2" transform="rotate(135 60 60)"/>
            <circle class="st-gauge__fill" id="ctx-fill" cx="60" cy="60" r="50" fill="none" stroke-width="10"
                    stroke-linecap="round" stroke-dasharray="0 314.2" transform="rotate(135 60 60)"/>
          </svg>
          <div class="ctx-gauge__label"><span class="ctx-gauge__pct" id="ctx-pct">0%</span><span class="muted" id="ctx-sub">&ndash;</span></div>
        </div>
        <div class="bar-row"><span><span data-lbl="main">main</span> experts in VRAM</span><span class="muted" id="slots-text">&ndash;</span></div>
        <div class="st-progress"><div class="st-progress__bar" id="slots-bar" style="width:0%"></div></div>
        <div class="bar-row"><span><span data-lbl="l2">lane2</span> context held</span><span class="muted" id="ctx-l2-text">&ndash;</span></div>
        <div class="st-progress" id="ctx-l2-progress"><div class="st-progress__bar" id="ctx-l2-bar" style="width:0%"></div></div>
        <div class="bar-row"><span><span data-lbl="l3">lane3</span> context held</span><span class="muted" id="ctx-l3-text">&ndash;</span></div>
        <div class="st-progress" id="ctx-l3-progress"><div class="st-progress__bar" id="ctx-l3-bar" style="width:0%"></div></div>
        <div class="bar-row"><span>System RAM</span><span class="muted" id="ram-text">&ndash;</span></div>
        <div class="st-progress" id="ram-progress"><div class="st-progress__bar" id="ram-bar" style="width:0%"></div></div>
        <div class="bar-row"><span>GPU temperature (max)</span><span class="muted" id="temp-text">&ndash;</span></div>
        <div class="st-progress" data-tone="warn"><div class="st-progress__bar" id="temp-bar" style="width:0%"></div></div>
      </div>
      <div class="st-card cc-card" id="cc-card">
        <div class="req-head"><span class="card-title">Conversation cache · <span data-lbl="main">main</span></span><span class="muted small" id="cc-sum"></span></div>
        <div class="cc-bars" id="cc-bars" hidden>
          <div><div class="bar-row"><span>Parked conversations</span><span class="muted" id="cc-slots-text">&ndash;</span></div>
            <div class="st-progress"><div class="st-progress__bar" id="cc-slots-bar" style="width:0%"></div></div></div>
          <div><div class="bar-row"><span>Their memory (RAM)</span><span class="muted" id="cc-mem-text">&ndash;</span></div>
            <div class="st-progress"><div class="st-progress__bar" id="cc-mem-bar" style="width:0%"></div></div></div>
        </div>
        <dl class="facts" id="cc-facts"></dl>
        <p class="muted small cc-note" id="cc-note"></p>
      </div>
    </div>

    <section class="st-card">
        <div class="req-head"><span class="card-title">Totals over a period</span>
          <span class="muted small" id="sum-span"></span></div>
        <div class="rangebar">
          <button class="st-btn st-btn--secondary" data-sum="1h">1h</button>
          <button class="st-btn st-btn--secondary on" data-sum="24h">24h</button>
          <button class="st-btn st-btn--secondary" data-sum="7d">7d</button>
          <button class="st-btn st-btn--secondary" data-sum="30d">30d</button>
          <button class="st-btn st-btn--secondary" data-sum="custom">custom</button>
          <span id="sum-custom" hidden>
            <input type="datetime-local" id="sum-from"> <input type="datetime-local" id="sum-to">
          </span>
        </div>
        <div class="table-wrap">
          <table class="st-table">
            <thead><tr><th>Engine</th><th class="num">Requests</th><th class="num" title="prompt tokens computed fresh (+ reused from cache)">Prompt read</th>
              <th class="num">Output</th><th class="num">Avg prefill t/s</th><th class="num">Avg decode t/s</th><th class="num" title="from per-request rows">Avg in / out</th>
              <th class="num" title="Auto-sessions disk cache: restores / saves, share of requests that came back from disk, average restore time, bytes parked">Disk cache</th></tr></thead>
            <tbody id="sum-body"><tr><td colspan="8" class="muted">loading…</td></tr></tbody>
          </table>
        </div>
    </section>
    <section class="st-card">
        <div class="req-head"><span class="card-title">Engine details</span><span class="muted small">read live from each lane: /metrics, its unit and launch preset</span></div>
        <div id="eng-facts"></div>
    </section>
  </div>
  <style>
    .rangebar { display:flex; gap:8px; align-items:center; margin:0 0 10px; flex-wrap:wrap; }
    .rangebar .st-btn.on { border-color: var(--st-accent); color: var(--st-accent); }
    .rangebar input { background: var(--st-surface); color: var(--st-ink); border: 1px solid var(--st-line); border-radius: 8px; padding: 4px 8px; }
    .state-card .card-title { font-size: .8rem; }
  </style>
"""

# ================================================================ hardware page
HW_JS = HELPERS + THEME_JS + r"""
// buffers per sensor key; live mode = last ~2 min, history mode = persisted minute rollups
const buf = new Map(), CAP = 120;
let HRANGE = "live", hwHist = null, hwGot = 0;
function push(k, v, t) {
  if (v == null || isNaN(v)) return;
  let a = buf.get(k);
  if (!a) { a = {x: [], y: []}; buf.set(k, a); }
  a.x.push(t != null ? t : Date.now()); a.y.push(Number(v));
  if (a.y.length > CAP) { a.x.shift(); a.y.shift(); }
}
// extract every chart series value out of one state/hw60 snapshot {hw, gpus, host}
function extract(s) {
  const hw = s.hw || [], gpus = s.gpus || [], host = s.host || {};
  const tempsOf = (name) => { for (const c of hw) if (c.chip === name) return c.temps.filter((t) => t.c >= 5); return []; };
  const kt = tempsOf("k10temp");
  const nv = tempsOf("nvme");
  const dimm = hw.filter((c) => c.chip.startsWith("spd")).flatMap((c) => c.temps).filter((t) => t.c >= 5);
  const out = {};
  out.cpuTctl = (kt.find((t) => t.label === "Tctl") || {}).c;
  out.cpuGHz = host.cpu_ghz;
  out.cpuTdie = host.cpu_temp;
  out.nvmeW = nv.length ? Math.max(...nv.map((t) => t.c)) : null;
  out.dimmW = dimm.length ? Math.max(...dimm.map((t) => t.c)) : null;
  for (const g of gpus) { out["gput" + g.index] = g["temperature.gpu"]; out["blower" + g.index] = g["fan.speed"] ?? 0; }
  for (const c of hw) {
    for (const f of (c.fans || [])) out["rpm" + c.chip + f.id] = f.rpm;
    for (const t of c.temps) if (t.c >= 5) out["sen" + c.chip + t.id] = t.c;
  }
  return out;
}
function fillHist(points) {
  buf.clear();
  for (const p of points) {
    const vals = extract(p);
    for (const k in vals) {
      const v = vals[k];
      if (v == null || isNaN(v)) continue;
      let a = buf.get(k);
      if (!a) { a = {x: [], y: []}; buf.set(k, a); }
      a.x.push(p.t * 1000); a.y.push(Number(v));
    }
  }
}
async function loadHw() {
  try {
    const r = await fetch(`api/hw?range=${HRANGE}`);
    if (r.ok) { hwHist = (await r.json()).points; hwGot = Date.now(); fillHist(hwHist); }
  } catch (e) { /* keep the old window */ }
}
document.querySelectorAll("[data-hrange]").forEach((b) => b.addEventListener("click", async () => {
  HRANGE = b.dataset.hrange;
  document.querySelectorAll("[data-hrange]").forEach((x) => x.classList.toggle("on", x === b));
  if (HRANGE === "live") buf.clear();
  else { hwHist = null; await loadHw(); }
  for (const k in HC) { try { HC[k].destroy(); } catch (e) { /* ignore */ } delete HC[k]; }
  if (lastHw) render(lastHw);
}));
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
// Board fan headers -> physical fans (map verified by eye 29.10, see docs/07).
const FAN_PHYS = { nct6799: {
  "2": ["AIO radiator fans", "CPU_FAN1 · top exhaust · fancontrol (CPU temp)"],
  "3": ["Rear exhaust 140", "CPU_FAN2 · fancontrol (CPU temp)"],
  "4": ["AIO pump", "AIO_PUMP · BIOS · fixed max speed"],
  "5": ["Front intake 3×140", "CHA_FAN · fancontrol · kept above rear"],
  "1": ["Spare channel", "no tach · bottom 2×P12 on CHA_FAN4 = BIOS only"] } };
function fanName(chip, id) { const p = (FAN_PHYS[chip] || {})[String(id)]; return p ? p[0] : `${chipName(chip)} · header ${id}`; }
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

  // feed the history buffers (live mode only; history mode fills from the persisted snapshots)
  if (HRANGE === "live") { const vals = extract(s); for (const k in vals) push(k, vals[k]); }

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
      const phys = (FAN_PHYS[c.chip] || {})[String(f.id)];
      fanCards.push(card(key, phys ? phys[0] : `${chipName(c.chip)} · header ${f.id}`, "activity",
        on ? `${fmt(f.rpm)}<small>RPM</small>` : "–", on ? (phys ? phys[1] : "") : (phys ? phys[1] : "nothing connected"), on ? "warn" : null, 36));
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
  for (const c of hw) for (const f of (c.fans || [])) chartFor("rpm" + c.chip + f.id, () => SC()[1], fanName(c.chip, f.id), 36);
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
    if (r.ok) { if (HRANGE !== "live" && (!hwHist || Date.now() - hwGot > 60000)) await loadHw(); render(await r.json()); }
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
    <div class="rangebar">
      <span class="muted small">charts:</span>
      <button class="st-btn st-btn--secondary on" data-hrange="live">Live 2 min</button>
      <button class="st-btn st-btn--secondary" data-hrange="1h">1 hour</button>
      <button class="st-btn st-btn--secondary" data-hrange="24h">1 day</button>
      <button class="st-btn st-btn--secondary" data-hrange="7d">7 days</button>
      <button class="st-btn st-btn--secondary" data-hrange="30d">30 days</button>
      <span class="muted small">history stored every minute, kept 30 days</span>
    </div>
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
.sess{{font-size:var(--st-fs-xs);white-space:nowrap}}
a.st-tab{{text-decoration:none}}
.updated-at{{font-size:var(--st-fs-xs);color:var(--st-ink-muted)}}
.st-pill[data-state="alert"] .st-pill__dot{{background:var(--st-danger);box-shadow:0 0 0 4px var(--st-danger-tint)}}
.hw-section{{width:100%}}
.hw-strip{{grid-template-columns:repeat(auto-fill,minmax(230px,1fr))}}
.hw-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:var(--st-s-3)}}
.hw-grid--wide{{grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:var(--st-s-4)}}
.hw-sensor{{margin-top:10px}}
.req-src{{padding:1px 6px;border-radius:5px}}
.req-src.m-main{{background:color-mix(in srgb,var(--st-accent) 20%,transparent)}}
.req-src.m-l2{{background:color-mix(in srgb,var(--st-info) 22%,transparent)}}
.req-src.m-l3{{background:color-mix(in srgb,var(--st-warn) 22%,transparent)}}
#req-body tr.m-main>td{{background:color-mix(in srgb,var(--st-accent) 6%,transparent)}}
#req-body tr.m-l2>td{{background:color-mix(in srgb,var(--st-info) 7%,transparent)}}
#req-body tr.m-l3>td{{background:color-mix(in srgb,var(--st-warn) 7%,transparent)}}
.eng-card.m-main{{border-top:3px solid var(--st-accent)}}
.eng-card.m-l2{{border-top:3px solid var(--st-info)}}
.eng-card.m-l3{{border-top:3px solid var(--st-warn)}}
.eng-card .card-title .req-src{{font-size:11px;font-weight:600;color:var(--st-ink)}}
.rangebar{{display:flex;gap:8px;align-items:center;margin:0 0 10px;flex-wrap:wrap}}
.rangebar .st-btn.on{{border-color:var(--st-accent);color:var(--st-accent)}}
.rangebar input{{background:var(--st-surface);color:var(--st-ink);border:1px solid var(--st-line);border-radius:8px;padding:4px 8px}}
.rangebar .st-btn.c-main.on{{border-color:var(--st-accent);color:var(--st-accent)}}
.rangebar .st-btn.c-l2.on{{border-color:var(--st-info);color:var(--st-info)}}
.rangebar .st-btn.c-l3.on{{border-color:var(--st-warn);color:var(--st-warn)}}
.req-head--f{{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;column-gap:12px}}
.req-head--f .card-title{{justify-self:start}}
.req-head--f .req-filters{{justify-self:center;margin:0}}
.req-head--f #req-all{{justify-self:end}}
.eng-tps{{display:flex;gap:32px;margin:10px 2px 8px}}
.eng-tps__m{{display:flex;flex-direction:column;gap:3px}}
.eng-tps .st-metric__value{{font-size:clamp(22px,2vw,var(--st-fs-stat))}}
.eng-tps__m small{{color:var(--st-ink-muted);font-size:var(--st-fs-xs);letter-spacing:.03em}}
.eng-tps .tps-last{{margin-top:2px;font-variant-numeric:tabular-nums}}
.row-half{{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}}
#eng-facts .bar-row{{font-size:var(--st-fs-xs);margin-bottom:4px}}
#eng-facts .facts{{grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:5px 8px;font-size:var(--st-fs-xs)}}
#eng-facts .facts dd{{flex-wrap:wrap}}
.speed-values{{display:flex;gap:var(--st-s-5);align-items:baseline}}
.speed-prefill .st-metric__value{{font-size:var(--st-fs-lg)}}
.mchart{{width:100%;height:112px;margin-top:2px}}
/* lane cards: equal thirds, the GPU/context line under the title, boot progress */
.eng-row{{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media (max-width:1100px){{.eng-row{{grid-template-columns:minmax(0,1fr)}}}}
.eng-card .state-card__head{{flex-wrap:wrap;row-gap:6px}}
.eng-sub{{font-size:var(--st-fs-xs);margin:-2px 0 8px;line-height:1.35}}
.eng-note{{font-size:var(--st-fs-xs);color:var(--st-warn);min-height:0;margin-top:4px}}
.eng-note:empty{{display:none}}
.eng-card.booting .st-progress__bar{{transition:width .9s linear}}
.req-src{{white-space:nowrap;display:inline-block}}
.st-table th{{white-space:nowrap}}
#eng-facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px 28px}}
#eng-facts .facts{{grid-template-columns:minmax(110px,auto) minmax(0,1fr) !important}}
#eng-facts .facts dd{{overflow-wrap:anywhere}}
.hchart{{width:100%;height:46px;margin-top:2px}}
{extra_css}
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

HIST_CSS = """
.sess{font-size:var(--st-fs-xs);white-space:nowrap}
#hq{background:var(--st-bg);color:var(--st-ink);border:1px solid var(--st-border);border-radius:var(--st-r-md);padding:6px 10px;font:inherit;font-size:var(--st-fs-sm);min-width:240px}
.h-actions{display:flex;gap:10px;align-items:center;margin-top:8px}
"""

HIST_BODY = """    <section class="view-section">
      <div class="st-card req-card" style="grid-column:1/-1">
        <div class="req-head req-head--f"><span class="card-title">Request history</span>
          <span class="rangebar">
            <button class="st-btn st-btn--secondary" data-hr="1h">1h</button>
            <button class="st-btn st-btn--secondary" data-hr="6h">6h</button>
            <button class="st-btn st-btn--secondary on" data-hr="24h">24h</button>
            <button class="st-btn st-btn--secondary" data-hr="7d">7d</button>
            <button class="st-btn st-btn--secondary" data-hr="30d">30d</button>
            <button class="st-btn st-btn--secondary" data-hr="all">all</button>
          </span>
          <span class="rangebar req-filters">
            <button class="st-btn st-btn--secondary on" data-he="all">all</button>
            <button class="st-btn st-btn--secondary c-main" data-he="main">main</button>
            <button class="st-btn st-btn--secondary c-l2" data-he="l2">lane2</button>
            <button class="st-btn st-btn--secondary c-l3" data-he="l3">lane3</button>
          </span>
          <input id="hq" type="search" placeholder="session / conversation key / status" autocomplete="off">
        </div>
        <div class="table-wrap all">
          <table class="st-table">
            <thead><tr><th>Time</th><th>Lane</th><th title="LiteLLM session id + conversation key">Session</th><th>Status</th>
              <th class="num">Prompt</th><th class="num">Reused</th><th class="num">Output</th>
              <th class="num">Prefill t/s</th><th class="num">Decode t/s</th>
              <th class="num" title="conversation restored from disk (auto-sessions) before this request: how long it took">Disk</th>
              <th class="num">Duration</th></tr></thead>
            <tbody id="h-body"><tr><td colspan="11" class="muted">loading…</td></tr></tbody>
          </table>
        </div>
        <div class="h-actions"><span class="muted small" id="h-count"></span>
          <button class="st-btn st-btn--secondary" id="h-more" style="visibility:hidden">200 more</button>
          <span class="muted small">rows are kept 90 days · updated <span id="h-updated">–</span></span></div>
      </div>
    </section>
"""

HIST_JS = HELPERS + THEME_JS + r"""
let HR = "24h", HE = "all", HQ = "", ROWS = [], TOTAL = 0;
const LBL = {main: "main", l2: "lane2", l3: "lane3"};
const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
               disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"]};
function render(){
  $("h-count").textContent = `${fmt(ROWS.length)} of ${fmt(TOTAL)} requests`;
  $("h-body").innerHTML = ROWS.length ? ROWS.map((r) => {
    const [cls, text] = badge[r.status] || [r.status && r.status !== "stop" ? "st-badge--error" : "", r.status || "–"];
    const t = new Date((r.ts || 0) * 1000).toLocaleString([], {month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23"});
    const hit = r.hit == null ? "–" : `${(r.hit * 100).toFixed(1)}%`;
    const sess = r.sess ? `<span title="session ${esc(r.sess)}${r.conv ? ` · conversation ${esc(r.conv)}` : ""}">${esc(r.sess.slice(0, 14))}${r.conv ? ` <span class="muted">c${esc(r.conv.slice(0, 6))}</span>` : ""}</span>` : "–";
    return `<tr class="m-${r.eng}"><td>${esc(t)}</td><td><span class="req-src m-${r.eng}">${LBL[r.eng] || esc(r.eng)}</span></td>
      <td class="sess">${sess}</td><td><span class="st-badge ${cls}">${esc(text)}</span></td>
      <td class="num">${fmt(r.prompt)}</td><td class="num">${fmt(r.reused)}</td><td class="num">${fmt(r.output)}</td>
      <td class="num">${fmt(r.prefill_tps)}</td><td class="num">${fmt(r.decode_tps, 1)}</td>
      <td class="num">${r.disk_ms != null ? `${fmt(r.disk_ms, 0)} ms` : "–"}</td><td class="num">${fmt(r.dur, 1)} s</td></tr>`;
  }).join("") : `<tr><td colspan="11" class="muted">nothing in this window</td></tr>`;
  $("h-more").style.visibility = ROWS.length < TOTAL ? "visible" : "hidden";
  $("h-updated").textContent = new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
}
async function load(more){
  try {
    const p = new URLSearchParams({range: HR, eng: HE, q: HQ, limit: 200, offset: more ? ROWS.length : 0});
    const s = await (await fetch(`api/requests?${p}`)).json();
    if (!more) ROWS = [];
    ROWS = ROWS.concat(s.rows || []);
    TOTAL = s.total || 0;
    render();
  } catch (e) { $("h-body").innerHTML = `<tr><td colspan="11" class="muted">history unavailable</td></tr>`; }
}
document.querySelectorAll("[data-hr]").forEach((b) => b.addEventListener("click", () => {
  HR = b.dataset.hr;
  document.querySelectorAll("[data-hr]").forEach((x) => x.classList.toggle("on", x === b));
  load(false);
}));
document.querySelectorAll("[data-he]").forEach((b) => b.addEventListener("click", () => {
  HE = b.dataset.he;
  document.querySelectorAll("[data-he]").forEach((x) => x.classList.toggle("on", x === b));
  load(false);
}));
let hqt; $("hq").addEventListener("input", () => { clearTimeout(hqt); hqt = setTimeout(() => { HQ = $("hq").value.trim(); load(false); }, 300); });
$("h-more").onclick = () => load(true);
load(false);
setInterval(() => { if (!ROWS.length || document.hasFocus()) load(false); }, 15000);
"""

(HERE / "index.html").write_text(page("Rig · monitor", "monitor", MON_BODY, MON_JS))
(HERE / "hw.html").write_text(page("Rig · hardware", "hw", HW_BODY, HW_JS))
(HERE / "history.html").write_text(page("Rig · history", "hist", HIST_BODY, HIST_JS, HIST_CSS))
print("wrote index.html + hw.html + history.html")
