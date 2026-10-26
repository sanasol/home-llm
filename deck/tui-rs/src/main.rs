//! NInfer monitor TUI (Ratatui edition).
//!
//! Polls /api/state from ninfer-monitor (:8090) once a second and renders a
//! graphical dashboard: huge double-scale pixel-font decode/prefill/KV numbers
//! with delta indicators, sparkline charts, gauge bars for GPU/VRAM/power/CPU/
//! RAM/disk/fan, model-weight load progress (2-row bar) and a recent-requests
//! table with mini-bars and colored hot columns.
//! Keys: q/Esc — quit, r — refresh now.
//!
//! Env: MONITOR_HOST (127.0.0.1), MONITOR_PORT (8090).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use chrono::Local;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Context, Line as CanvasLine, Points};
use ratatui::widgets::{Block, BorderType, Borders, Cell, Paragraph, Row, Table};
use ratatui::Frame;
use ratatui::Terminal;
use serde_json::Value;

const CYAN: Color = Color::Cyan;
const GREEN: Color = Color::Green;
const YELLOW: Color = Color::Yellow;
const RED: Color = Color::Red;
const MAGENTA: Color = Color::Magenta;
const DIM: Color = Color::DarkGray;
const GRAY: Color = Color::Gray;

// 5x7 pixel font for the big numbers (bit 4 = leftmost column)
const FONT: &[(&str, [u8; 7])] = &[
    ("0", [14, 17, 19, 21, 25, 17, 14]),
    ("1", [4, 12, 4, 4, 4, 4, 14]),
    ("2", [14, 17, 1, 6, 8, 16, 31]),
    ("3", [14, 17, 1, 7, 1, 17, 14]),
    ("4", [2, 6, 10, 18, 31, 2, 2]),
    ("5", [31, 16, 30, 1, 1, 17, 14]),
    ("6", [6, 8, 16, 30, 17, 17, 14]),
    ("7", [31, 1, 1, 4, 8, 8, 8]),
    ("8", [14, 17, 17, 14, 17, 17, 14]),
    ("9", [14, 17, 17, 15, 1, 2, 12]),
    (".", [0, 0, 0, 0, 0, 0, 14]),
    ("k", [8, 10, 12, 8, 12, 10, 10]),
    ("-", [0, 0, 0, 31, 0, 0, 0]),
];

fn glyph(ch: char) -> Option<[u8; 7]> {
    FONT.iter().find(|(c, _)| c == &ch.to_string()).map(|(_, r)| *r)
}

/// Double-scale pixel font: each pixel = 2 cells wide, each row = 2 lines tall.
/// Yields 14 lines, each glyph 10 cols wide.
fn big_lines(text: &str) -> Vec<Line> {
    let rows: Vec<String> = (0..7)
        .map(|r| {
            let mut line = String::new();
            for ch in text.chars() {
                if let Some(bits) = glyph(ch) {
                    for c in 0..5 {
                        if bits[r] & (16 >> c) != 0 {
                            line.push_str("██");
                        } else {
                            line.push_str("  ");
                        }
                    }
                    line.push_str("  ");
                } else {
                    line.push_str("      ");
                }
            }
            line
        })
        .collect();
    (0..14).map(|r| Line::from(rows[r / 2].clone())).collect()
}

fn big_val(v: Option<f64>) -> String {
    match v {
        None => "-".into(),
        Some(x) if x >= 1000.0 => format!("{:.1}k", x / 1000.0),
        Some(x) => format!("{}", x.round() as i64),
    }
}

fn fm(v: Option<f64>, d: u32) -> String {
    match v {
        None => "-".into(),
        Some(x) => format!("{:.*}", d as usize, x),
    }
}

fn extract(s: &str, key: &str) -> String {
    let pat = format!("{}=\"", key);
    s.find(&pat)
        .map(|i| {
            let rest = &s[i + pat.len()..];
            rest.find('"').map(|j| rest[..j].to_string()).unwrap_or_default()
        })
        .unwrap_or_default()
}

// -------------------------------------------------------------- sparks -----

fn minibar(pct: f64, n: usize) -> String {
    let k = ((n as f64 * pct.clamp(0.0, 100.0)) / 100.0).round() as usize;
    format!("{}{}", "█".repeat(k), "░".repeat(n.saturating_sub(k)))
}

// ---------------------------------------------------------------- state ----

struct ReqRow {
    time: String,
    api: String,
    prompt: f64,
    cached: f64,
    output: f64,
    ttft: f64,
    pre: f64,
    dec: f64,
    acc: Option<f64>,
    total: f64,
}

struct Stats {
    ok: bool,
    err: String,
    model: String,
    spec: String,
    ctx: String,
    ready: bool,
    service: String,
    updated: f64,
    running: Option<f64>,
    waiting: Option<f64>,
    paused: Option<f64>,
    maxc: Option<f64>,
    kv_pct: Option<f64>,
    kv_sub: String,
    cache_pct: Option<f64>,
    cache_sub: String,
    acc_pct: Option<f64>,
    acc_sub: String,
    ttft: Option<f64>,
    done: i64,
    failed: i64,
    preemptions: i64,
    gpu: Option<f64>,
    power: Option<f64>,
    power_limit: Option<f64>,
    temp: Option<f64>,
    vram: Option<f64>,
    vram_txt: String,
    pcie: String,
    fan: Option<f64>,
    clock: Option<f64>,
    cpu: Option<f64>,
    cpu_sub: String,
    ram: Option<f64>,
    ram_txt: String,
    ram_sub: String,
    disk: Option<f64>,
    disk_txt: String,
    disk_sub: String,
    decode: Option<f64>,
    prefill: Option<f64>,
    dec_hist: Vec<f64>,
    pre_hist: Vec<f64>,
    load_active: bool,
    load_final: bool,
    load_pct: f64,
    load_sub: String,
    rows: Vec<ReqRow>,
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            ok: false,
            err: "connecting…".into(),
            model: "?".into(),
            spec: String::new(),
            ctx: String::new(),
            ready: false,
            service: String::new(),
            updated: 0.0,
            running: None,
            waiting: None,
            paused: None,
            maxc: None,
            kv_pct: None,
            kv_sub: "-".into(),
            cache_pct: None,
            cache_sub: "-".into(),
            acc_pct: None,
            acc_sub: "-".into(),
            ttft: None,
            done: 0,
            failed: 0,
            preemptions: 0,
            gpu: None,
            power: None,
            power_limit: None,
            temp: None,
            vram: None,
            vram_txt: "-".into(),
            pcie: "-".into(),
            fan: None,
            clock: None,
            cpu: None,
            cpu_sub: "-".into(),
            ram: None,
            ram_txt: "-".into(),
            ram_sub: "-".into(),
            disk: None,
            disk_txt: "-".into(),
            disk_sub: "-".into(),
            decode: None,
            prefill: None,
            dec_hist: Vec::new(),
            pre_hist: Vec::new(),
            load_active: false,
            load_final: false,
            load_pct: 0.0,
            load_sub: String::new(),
            rows: Vec::new(),
        }
    }
}

fn num(v: &Value) -> Option<f64> {
    v.as_f64()
}
fn str_(v: &Value) -> String {
    v.as_str().unwrap_or("").to_string()
}

impl Stats {
    fn parse(v: &Value) -> Self {
        let mut s = Self::default();
        s.ok = true;
        let m = v.get("ninfer").unwrap_or(&Value::Null);
        let g = v.get("gpu").unwrap_or(&Value::Null);
        let h = v.get("host").unwrap_or(&Value::Null);
        let h2 = v.get("history").unwrap_or(&Value::Null);

        s.ready = m.get("ninfer_engine_ready").and_then(num) == Some(1.0);
        s.service = str_(v.get("service").unwrap_or(&Value::Null));
        s.updated = v.get("updated").and_then(num).unwrap_or(0.0);
        if let Some(obj) = m.as_object() {
            for (k, _mv) in obj.iter() {
                if let Some(rest) = k.strip_prefix("ninfer_model_info") {
                    s.model = extract(rest, "model_name");
                    s.spec = extract(rest, "speculative_backend");
                }
            }
        }
        if let Some(ctx) = m.get("ninfer_max_context_tokens").and_then(num) {
            if ctx > 1024.0 {
                s.ctx = format!(" · {:.0}K ctx", ctx / 1024.0);
            }
        }

        s.running = m.get("ninfer_requests_running").and_then(num);
        s.waiting = m.get("ninfer_requests_waiting").and_then(num);
        s.paused = m.get("ninfer_requests_paused").and_then(num);
        s.maxc = m.get("ninfer_max_concurrency").and_then(num);

        let kv_u = m.get(r#"ninfer_device_kv_used_page_groups{pool_id="0"}"#).and_then(num);
        let kv_c = m.get(r#"ninfer_device_kv_capacity_page_groups{pool_id="0"}"#).and_then(num);
        s.kv_pct = match (kv_u, kv_c) {
            (Some(u), Some(c)) if c > 0.0 => Some(100.0 * u / c),
            _ => None,
        };
        if let (Some(u), Some(c)) = (kv_u, kv_c) {
            if let Some(cap_tok) = v.get("kv_capacity_tokens").and_then(num) {
                s.kv_sub = format!(
                    "≈{:.0}K / {:.0}K tok",
                    (cap_tok * u / c) / 1000.0,
                    cap_tok / 1000.0
                );
            }
        }

        let prompt = m.get("ninfer_prompt_tokens_total").and_then(num);
        let cached = m.get("ninfer_prompt_tokens_cached_total").and_then(num);
        if let (Some(p), Some(c)) = (prompt, cached) {
            if p > 0.0 {
                s.cache_pct = Some(100.0 * c / p);
                s.cache_sub = format!(
                    "{}K / {}K tok",
                    (c / 1000.0).round(),
                    (p / 1000.0).round()
                );
            }
        }
        let drafted = m.get("ninfer_spec_decode_draft_tokens_total").and_then(num);
        let accepted = m.get("ninfer_spec_decode_accepted_tokens_total").and_then(num);
        if let (Some(d), Some(a)) = (drafted, accepted) {
            if d > 0.0 {
                s.acc_pct = Some(100.0 * a / d);
                s.acc_sub = format!("{} tok", (a / 1000.0).round());
            }
        }
        let ttft_n = m.get("ninfer_time_to_first_token_seconds_count").and_then(num);
        let ttft_s = m.get("ninfer_time_to_first_token_seconds_sum").and_then(num);
        s.ttft = match (ttft_n, ttft_s) {
            (Some(n), Some(x)) if n > 0.0 => Some(x / n),
            _ => None,
        };
        s.done = m
            .get(r#"ninfer_requests_total{outcome="completed"}"#)
            .and_then(num)
            .unwrap_or(0.0) as i64;
        s.failed = (m
            .get(r#"ninfer_requests_total{outcome="failed"}"#)
            .and_then(num)
            .unwrap_or(0.0)
            + m
                .get(r#"ninfer_requests_total{outcome="rejected"}"#)
                .and_then(num)
                .unwrap_or(0.0)) as i64;
        s.preemptions = m.get("ninfer_preemptions_total").and_then(num).unwrap_or(0.0) as i64;

        s.gpu = g.get("utilization.gpu").and_then(num);
        s.power = g.get("power.draw").and_then(num);
        s.power_limit = g.get("power.limit").and_then(num);
        s.temp = g.get("temperature.gpu").and_then(num);
        let mu = g.get("memory.used").and_then(num);
        let mt = g.get("memory.total").and_then(num);
        s.vram = match (mu, mt) {
            (Some(u), Some(t)) if t > 0.0 => Some(100.0 * u / t),
            _ => None,
        };
        s.vram_txt = format!(
            "{}/{} GiB",
            fm(mu.map(|x| x / 1024.0), 1),
            fm(mt.map(|x| x / 1024.0), 1)
        );
        s.pcie = format!(
            "Gen{} x{}",
            fm(g.get("pcie.link.gen.current").and_then(num), 0),
            fm(g.get("pcie.link.width.current").and_then(num), 0)
        );
        s.fan = g.get("fan.speed").and_then(num);
        s.clock = g.get("clocks.sm").and_then(num);

        if let Some(arr) = h2.as_array() {
            let mut d: Vec<f64> = Vec::new();
            let mut p: Vec<f64> = Vec::new();
            for pt in arr {
                if let Some(x) = pt.get("decode_tps").and_then(num) {
                    d.push(x);
                }
                if let Some(x) = pt.get("prefill_tps").and_then(num) {
                    p.push(x);
                }
            }
            s.decode = d.last().copied();
            s.prefill = p.last().copied();
            s.cpu = arr.iter().rev().find_map(|pt| pt.get("cpu_pct").and_then(num));
            s.dec_hist = d.into_iter().take(60).collect();
            s.pre_hist = p.into_iter().take(60).collect();
        }

        s.cpu_sub = format!("load {}", fm(h.get("load1").and_then(num), 2));
        let mtot = h.get("mem_total").and_then(num).unwrap_or(0.0);
        let mav = h.get("mem_available").and_then(num).unwrap_or(0.0);
        if mtot > 0.0 {
            let used = (mtot - mav) / 1073741824.0;
            s.ram = Some(100.0 * (mtot - mav) / mtot);
            s.ram_txt = format!(
                "{}/{} GiB",
                fm(Some(used), 1),
                fm(Some(mtot / 1073741824.0), 1)
            );
            s.ram_sub = format!("avail {} GiB", fm(Some(mav / 1073741824.0), 1));
        }
        let dtot = h.get("disk_total").and_then(num);
        let dav = h.get("disk_avail").and_then(num);
        if let (Some(t), Some(a)) = (dtot, dav) {
            if t > 0.0 {
                let gib = 1073741824.0;
                s.disk = Some(100.0 * (t - a) / t);
                s.disk_txt = format!(
                    "{}/{} GiB",
                    fm(Some((t - a) / gib), 0),
                    fm(Some(t / gib), 0)
                );
            }
        }
        if let Some(arr) = h2.as_array() {
            if let Some(pt) = arr.last() {
                s.disk_sub = format!(
                    "r {} · w {} MB/s",
                    fm(pt.get("disk_r_mbps").and_then(num), 2),
                    fm(pt.get("disk_w_mbps").and_then(num), 2)
                );
            }
        }

        let ld = v.get("model_load").unwrap_or(&Value::Null);
        match str_(ld.get("phase").unwrap_or(&Value::Null)).as_str() {
            "loading" => {
                s.load_active = true;
                s.load_pct = ld.get("pct").and_then(num).unwrap_or(0.0);
                s.load_sub = format!(
                    "{:.1}/{:.1} GiB · {} MiB/s · ETA {}",
                    ld.get("loaded_gib").and_then(num).unwrap_or(0.0),
                    ld.get("total_gib").and_then(num).unwrap_or(0.0),
                    fm(ld.get("speed_mib_s").and_then(num), 0),
                    str_(ld.get("eta_text").unwrap_or(&Value::Null))
                );
            }
            "finalizing" => {
                s.load_active = true;
                s.load_final = true;
                s.load_pct = 100.0;
                s.load_sub = "finalizing: CUDA graphs, host context".into();
            }
            _ => {}
        }

        for r in v
            .get("requests")
            .and_then(Value::as_array)
            .unwrap_or(&vec![])
        {
            let mut api = str_(r.get("api").unwrap_or(&Value::Null));
            if api.len() > 13 {
                let cut = api
                    .char_indices()
                    .nth(13)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                api = format!("{}…", &api[..cut]);
            }
            s.rows.push(ReqRow {
                time: str_(r.get("ts").unwrap_or(&Value::Null))
                    .chars()
                    .skip(11)
                    .take(5)
                    .collect(),
                api,
                prompt: r.get("prompt").and_then(num).unwrap_or(0.0),
                cached: r.get("cache_pct").and_then(num).unwrap_or(0.0),
                output: r.get("output").and_then(num).unwrap_or(0.0),
                ttft: r.get("ttft").and_then(num).unwrap_or(0.0),
                pre: r.get("prefill_tps").and_then(num).unwrap_or(0.0),
                dec: r.get("decode_tps").and_then(num).unwrap_or(0.0),
                acc: r.get("accept_pct").and_then(num),
                total: r.get("total").and_then(num).unwrap_or(0.0),
            });
        }
        if s.rows.len() > 14 {
            s.rows.truncate(14);
        }
        s
    }
}

// -------------------------------------------------------------- fetching ---

fn fetch() -> Result<Value, String> {
    let host = std::env::var("MONITOR_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("MONITOR_PORT")
        .unwrap_or_else(|_| "8090".into())
        .parse()
        .map_err(|_| "bad MONITOR_PORT".to_string())?;
    let addr = format!("{}:{}", host, port)
        .parse::<std::net::SocketAddr>()
        .map_err(|e| e.to_string())?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))
        .map_err(|e| format!("connect {}: {}", host, e))?;
    stream
        .write_all(
            format!("GET /api/state HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n")
                .as_bytes(),
        )
        .map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&buf).into_owned();
    let body = text
        .find("\r\n\r\n")
        .map(|i| &text[i + 4..])
        .unwrap_or("");
    serde_json::from_str(body).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------- drawing --

fn threshold_color(pct: Option<f64>, warn: f64, crit: f64) -> Color {
    match pct {
        Some(p) if p >= crit => RED,
        Some(p) if p >= warn => YELLOW,
        _ => GREEN,
    }
}

fn delta_of(hist: &[f64]) -> (i64, Color) {
    if hist.len() >= 2 {
        let d = (hist[hist.len() - 1] - hist[hist.len() - 2]).round() as i64;
        (
            d,
            if d > 0 {
                GREEN
            } else if d < 0 {
                RED
            } else {
                DIM
            },
        )
    } else {
        (0, DIM)
    }
}

/// label(12) + bar(24) + value(15) + sub
fn gauge_line(label: &str, pct: Option<f64>, value: &str, sub: &str, color: Color) -> Line<'static> {
    let bar = match pct {
        None => "░".repeat(24),
        Some(p) => minibar(p, 24),
    };
    Line::from(vec![
        Span::styled(format!("{:<12}", label), Style::default().fg(GRAY)),
        Span::styled(bar, Style::default().fg(color)),
        Span::raw("  "),
        Span::styled(
            value.to_string(),
            Style::default().add_modifier(Modifier::BOLD).fg(color),
        ),
        Span::raw("  "),
        Span::styled(sub.to_string(), Style::default().fg(DIM)),
    ])
}

fn bordered(title: &str, color: Color, border: Color) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border))
        .title(Line::from(Span::styled(
            format!(" {} ", title),
            Style::default().add_modifier(Modifier::BOLD).fg(color),
        )))
}

fn panel(title: &str, color: Color, border: Color, lines: Vec<Line>, area: Rect, f: &mut Frame) {
    let block = bordered(title, color, border);
    let inner = block.inner(area);
    f.render_widget(block, area);
    f.render_widget(Paragraph::new(lines), inner);
}

/// Real chart (braille canvas): filled area + curve + peak line for one series.
fn draw_chart(
    f: &mut Frame,
    title: &str,
    hist: &[f64],
    current: Option<f64>,
    color: Color,
    area: Rect,
) {
    let data: Vec<f64> = if hist.len() > 60 {
        hist[hist.len() - 60..].to_vec()
    } else {
        hist.to_vec()
    };
    let n = data.len();
    let peak = data.iter().cloned().fold(0.0, f64::max);
    let avg = if n > 0 { data.iter().sum::<f64>() / n as f64 } else { 0.0 };
    let title_line = Line::from(vec![
        Span::styled(
            format!(" {} ", title),
            Style::default().add_modifier(Modifier::BOLD).fg(color),
        ),
        Span::styled(
            format!(
                "now {}  ·  peak {}  ·  avg {}   ({} s)",
                big_val(current),
                big_val(if peak > 0.0 { Some(peak) } else { None }),
                big_val(if avg > 0.0 { Some(avg) } else { None }),
                n
            ),
            Style::default().fg(DIM),
        ),
    ]);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(color))
        .title(title_line);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if n < 2 || inner.width < 8 || inner.height < 3 {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "  collecting history…",
                Style::default().fg(DIM),
            ))),
            inner,
        );
        return;
    }
    let h = inner.height as usize;
    let ymax = (peak * 1.15).max(1.0);
    let ystep = ymax / (h as f64 - 1.0);
    let mut fill: Vec<(f64, f64)> = Vec::new();
    for (i, v) in data.iter().enumerate() {
        let x = i as f64;
        let kmax = (*v / ystep).floor() as i64;
        for k in 0..=kmax {
            fill.push((x, k as f64 * ystep));
        }
    }
    let mut segs: Vec<CanvasLine> = Vec::new();
    for i in 0..n - 1 {
        segs.push(CanvasLine::new(i as f64, data[i], (i + 1) as f64, data[i + 1], color));
    }
    let peak_line = CanvasLine::new(0.0, peak, (n - 1) as f64, peak, Color::DarkGray);
    let canvas = Canvas::default()
        .x_bounds([0.0, (n - 1) as f64])
        .y_bounds([0.0, ymax])
        .paint(move |ctx: &mut Context| {
            ctx.draw(&Points {
                coords: &fill,
                color: Color::DarkGray,
            });
            for s in &segs {
                ctx.draw(s);
            }
            ctx.draw(&peak_line);
        });
    f.render_widget(canvas, inner);
}

fn draw(f: &mut Frame, st: &Stats) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // header
            Constraint::Length(17), // big stat blocks
            Constraint::Length(2), // load row
            Constraint::Length(9), // decode / prefill charts
            Constraint::Length(10), // LLM + hardware
            Constraint::Min(9), // requests
            Constraint::Length(1), // status bar
        ])
        .split(area);

    // ---- header ----
    let stale = st.ok && st.updated > 0.0 && Local::now().timestamp() as f64 - st.updated > 5.0;
    let (svc_txt, svc_color) = if !st.ok {
        ("DOWN", RED)
    } else if stale {
        ("STALE", RED)
    } else if st.ready {
        ("READY", GREEN)
    } else if matches!(st.service.as_str(), "active" | "activating") {
        ("LOADING", YELLOW)
    } else {
        ("DOWN", RED)
    };
    let title = Line::from(vec![
        Span::styled(
            " DECK 5090 · NINFER ",
            Style::default().add_modifier(Modifier::BOLD).fg(CYAN),
        ),
        Span::styled(
            format!(
                "{}{}{}",
                st.model,
                if st.spec.is_empty() {
                    String::new()
                } else {
                    format!(" · {}", st.spec)
                },
                st.ctx
            ),
            Style::default().fg(MAGENTA),
        ),
        Span::raw("   "),
        Span::styled(
            format!("[{}]", svc_txt),
            Style::default().add_modifier(Modifier::BOLD).fg(svc_color),
        ),
        Span::styled(
            Local::now().format("  %H:%M:%S").to_string(),
            Style::default().fg(DIM),
        ),
    ]);
    // gradient strip under the header
    let w = chunks[0].width as usize;
    let mut s1 = String::new();
    let mut s2 = String::new();
    let ramp: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    for i in 0..w {
        let ch = ramp[(i * 8).div_ceil(w).min(7)];
        if i < w * 2 / 3 {
            s1.push(ch);
        } else {
            s2.push(ch);
        }
    }
    let strip = Line::from(vec![
        Span::styled(s1, Style::default().fg(CYAN)),
        Span::styled(s2, Style::default().fg(DIM)),
    ]);
    f.render_widget(Paragraph::new(vec![title, strip]), chunks[0]);

    // ---- big stat blocks (double-scale 10x14 digits) ----
    let decode_val = big_val(st.decode);
    let prefill_val = big_val(st.prefill);
    let kv_val = big_val(st.kv_pct);
    let (d_dec, c_dec) = delta_of(&st.dec_hist);
    let (d_pre, c_pre) = delta_of(&st.pre_hist);
    fn stat_summary(hist: &[f64]) -> (Option<f64>, Option<f64>) {
        if hist.is_empty() {
            (None, None)
        } else {
            let peak = hist.iter().cloned().fold(f64::MIN, f64::max);
            let avg = hist.iter().sum::<f64>() / hist.len() as f64;
            (Some(peak), Some(avg))
        }
    }
    let (dec_peak, dec_avg) = stat_summary(&st.dec_hist);
    let (pre_peak, pre_avg) = stat_summary(&st.pre_hist);
    let blocks: Vec<(&str, &str, Color, Option<f64>, Option<f64>, Option<f64>, i64, Color)> =
        vec![
            (
                "DECODE · tok/s",
                &decode_val,
                if st.decode.is_some() {
                    GREEN
                } else {
                    DIM
                },
                st.decode,
                dec_peak,
                dec_avg,
                d_dec,
                c_dec,
            ),
            (
                "PREFILL · tok/s",
                &prefill_val,
                if st.prefill.is_some() {
                    MAGENTA
                } else {
                    DIM
                },
                st.prefill,
                pre_peak,
                pre_avg,
                d_pre,
                c_pre,
            ),
            (
                "KV POOL · %",
                &kv_val,
                if st.kv_pct.is_some() {
                    CYAN
                } else {
                    DIM
                },
                st.kv_pct,
                Some(100.0),
                None,
                0,
                DIM,
            ),
        ];
    let stat_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(chunks[1]);
    for (i, (title, value, color, cur, peak, avg, delta, dcolor)) in blocks.iter().enumerate() {
        let area = stat_chunks[i];
        let block = bordered(title, *color, *color);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let inner2 = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(14), Constraint::Length(1)])
            .split(inner);
        f.render_widget(
            Paragraph::new(big_lines(value))
                .style(Style::default().add_modifier(Modifier::BOLD).fg(*color))
                .alignment(Alignment::Center),
            inner2[0],
        );
        // bottom row: current/max progress bar + avg + delta
        let bar = match (*cur, *peak) {
            (Some(c), Some(p)) if p > 0.0 => minibar(100.0 * c / p, 14),
            _ => "░".repeat(14),
        };
        let mut spans: Vec<Span> = vec![
            Span::styled(bar, Style::default().fg(*color)),
            Span::raw(" "),
            Span::styled(
                value.to_string(),
                Style::default().add_modifier(Modifier::BOLD).fg(*color),
            ),
            Span::styled(" / ", Style::default().fg(DIM)),
            Span::styled(big_val(*peak), Style::default().fg(GRAY)),
            Span::styled(format!(" avg {}", big_val(*avg)), Style::default().fg(DIM)),
        ];
        if *delta != 0 {
            spans.push(Span::styled(
                format!(" Δ {:+}", delta),
                Style::default().add_modifier(Modifier::BOLD).fg(*dcolor),
            ));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), inner2[1]);
    }

    // ---- model load (2 rows) ----
    let barw = 40;
    if st.load_active {
        let color = if st.load_final {
            GREEN
        } else {
            YELLOW
        };
        let label = if st.load_final {
            " WEIGHTS READY — "
        } else {
            " LOADING WEIGHTS "
        };
        let line1 = if st.load_final {
            Line::from(vec![
                Span::styled(label, Style::default().add_modifier(Modifier::BOLD).fg(color)),
                Span::styled("█".repeat(barw), Style::default().fg(color)),
                Span::styled(format!("  {}", st.load_sub), Style::default().fg(color)),
            ])
        } else {
            Line::from(vec![
                Span::styled(label, Style::default().add_modifier(Modifier::BOLD).fg(color)),
                Span::styled(minibar(st.load_pct, barw), Style::default().fg(color)),
                Span::styled(
                    format!("  {:4.1}%", st.load_pct),
                    Style::default().add_modifier(Modifier::BOLD).fg(color),
                ),
                Span::styled(format!("  {}", st.load_sub), Style::default().fg(color)),
            ])
        };
        let line2 = Line::from(Span::styled(
            format!("  {}   weights → VRAM", "░".repeat(barw)),
            Style::default().fg(DIM),
        ));
        f.render_widget(Paragraph::new(vec![line1, line2]), chunks[2]);
    }

    // ---- decode / prefill charts (real canvas charts) ----
    let chart_halves = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[3]);
    draw_chart(
        f,
        "DECODE · tok/s",
        &st.dec_hist,
        st.decode,
        GREEN,
        chart_halves[0],
    );
    draw_chart(
        f,
        "PREFILL · tok/s",
        &st.pre_hist,
        st.prefill,
        MAGENTA,
        chart_halves[1],
    );

    // ---- LLM / HARDWARE ----
    let req_color = if st.running.unwrap_or(0.0) > 0.0 {
        GREEN
    } else {
        DIM
    };
    let engine_txt = if st.ready { "ready" } else { "starting" };
    let llm_lines = vec![
        gauge_line(
            "requests",
            st.maxc.map(|m| 100.0 * st.running.unwrap_or(0.0) / m),
            &format!("{}/{}", fm(st.running, 0), fm(st.maxc, 0)),
            &format!("wait {} · paused {}", fm(st.waiting, 0), fm(st.paused, 0)),
            req_color,
        ),
        gauge_line(
            "KV pool",
            st.kv_pct,
            &(fm(st.kv_pct, 0) + "%"),
            &st.kv_sub,
            CYAN,
        ),
        gauge_line(
            "cache hits",
            st.cache_pct,
            &(fm(st.cache_pct, 1) + "%"),
            &st.cache_sub,
            GREEN,
        ),
        gauge_line(
            "draft accept",
            st.acc_pct,
            &(fm(st.acc_pct, 1) + "%"),
            &st.acc_sub,
            GREEN,
        ),
        gauge_line(
            "TTFT avg",
            st.ttft.map(|t| (t / 5.0 * 100.0).min(100.0)),
            &(fm(st.ttft, 2) + " s"),
            &format!("{} done · {} fail", st.done, st.failed),
            threshold_color(st.ttft.map(|t| t * 20.0), 100.0, 200.0),
        ),
        Line::from(vec![
            Span::styled("preempts  ", Style::default().fg(GRAY)),
            Span::styled(format!("{}", st.preemptions), Style::default().fg(YELLOW)),
            Span::styled("  (requests killed mid-flight)", Style::default().fg(DIM)),
        ]),
        gauge_line(
            "engine",
            Some(if st.ready { 100.0 } else { 50.0 }),
            engine_txt,
            &st.model,
            if st.ready {
                GREEN
            } else {
                YELLOW
            },
        ),
        Line::from(Span::styled(
            format!(
                "  service {} · updated {}",
                st.service,
                Local::now().format("%H:%M:%S")
            ),
            Style::default().fg(DIM),
        )),
    ];
    let gpu_sub = format!(
        "{} / {} W · {} °C",
        fm(st.power, 0),
        fm(st.power_limit, 0),
        fm(st.temp, 0)
    );
    let power_pct = st
        .power
        .zip(st.power_limit)
        .map(|(a, b)| (100.0 * a / b).min(100.0));
    let hw_lines = vec![
        gauge_line(
            "GPU",
            st.gpu,
            &(fm(st.gpu, 0) + "%"),
            &gpu_sub,
            threshold_color(st.gpu, 90.0, 99.0),
        ),
        gauge_line(
            "VRAM",
            st.vram,
            &st.vram_txt,
            &st.pcie,
            threshold_color(st.vram, 85.0, 95.0),
        ),
        gauge_line(
            "power",
            power_pct,
            &(fm(st.power, 0) + " W"),
            &format!("limit {} W", fm(st.power_limit, 0)),
            threshold_color(power_pct, 80.0, 99.0),
        ),
        gauge_line(
            "CPU",
            st.cpu,
            &(fm(st.cpu, 1) + "%"),
            &st.cpu_sub,
            threshold_color(st.cpu, 80.0, 99.0),
        ),
        gauge_line(
            "RAM",
            st.ram,
            &st.ram_txt,
            &st.ram_sub,
            threshold_color(st.ram, 85.0, 95.0),
        ),
        gauge_line(
            "disk",
            st.disk,
            &st.disk_txt,
            &st.disk_sub,
            threshold_color(st.disk, 90.0, 99.0),
        ),
        gauge_line(
            "fan",
            st.fan,
            &(fm(st.fan, 0) + "%"),
            &format!("SM {} MHz", fm(st.clock, 0)),
            GREEN,
        ),
        Line::from(Span::styled(
            format!(
                "  service {} · engine {}",
                st.service,
                if st.ready { "ready" } else { "starting" }
            ),
            Style::default().fg(DIM),
        )),
    ];
    let main = chunks[4];
    if main.width >= 100 {
        let halves = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(main);
        panel("LLM", CYAN, CYAN, llm_lines, halves[0], f);
        panel("HARDWARE", GREEN, GREEN, hw_lines, halves[1], f);
    } else {
        let stacked = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(10), Constraint::Length(10)])
            .split(main);
        panel("LLM", CYAN, CYAN, llm_lines, stacked[0], f);
        panel("HARDWARE", GREEN, GREEN, hw_lines, stacked[1], f);
    }

    // ---- recent requests (mini-bars + colored hot columns) ----
    let header = Row::new(
        ["time", "api", "prompt", "cached", "out", "ttft", "pre t/s", "dec t/s", "acc", "total"]
            .into_iter()
            .map(|h| {
                Cell::from(Span::styled(
                    h.to_string(),
                    Style::default().add_modifier(Modifier::BOLD).fg(GRAY),
                ))
            }),
    );
    let rows: Vec<Row> = st
        .rows
        .iter()
        .map(|r| {
            let dec_color = if r.dec >= 80.0 {
                if r.dec >= 300.0 {
                    GREEN
                } else {
                    YELLOW
                }
            } else {
                DIM
            };
            let acc_color = match r.acc {
                Some(a) if a >= 60.0 => GREEN,
                Some(a) if a >= 40.0 => YELLOW,
                _ => DIM,
            };
            let cached_cell = format!("{:8}  {}%", minibar(r.cached, 8), r.cached.round() as i64);
            let acc_cell = match r.acc {
                Some(a) => format!("{:8}  {}%", minibar(a, 8), a.round() as i64),
                None => "-".into(),
            };
            Row::new(vec![
                Cell::from(r.time.clone()),
                Cell::from(r.api.clone()),
                Cell::from(format!("{}", r.prompt.round() as i64)),
                Cell::from(Span::styled(cached_cell, Style::default().fg(GREEN))),
                Cell::from(format!("{}", r.output.round() as i64)),
                Cell::from(format!("{:.2}", r.ttft)),
                Cell::from(format!("{}", r.pre.round() as i64)),
                Cell::from(Span::styled(
                    format!("{}", r.dec.round() as i64),
                    Style::default().add_modifier(Modifier::BOLD).fg(dec_color),
                )),
                Cell::from(Span::styled(acc_cell, Style::default().fg(acc_color))),
                Cell::from(format!("{:.1}", r.total)),
            ])
        })
        .collect();
    let table = Table::new(
        rows,
        [
            Constraint::Length(6),
            Constraint::Length(14),
            Constraint::Length(9),
            Constraint::Length(14),
            Constraint::Length(7),
            Constraint::Length(6),
            Constraint::Length(7),
            Constraint::Length(9),
            Constraint::Length(14),
            Constraint::Min(6),
        ],
    )
    .header(header)
    .block(bordered("RECENT REQUESTS", YELLOW, GRAY));
    f.render_widget(table, chunks[5]);

    // ---- status bar ----
    let dotc = if st.ok { GREEN } else { RED };
    let left = if st.ok {
        format!(
            "monitor ok · {} requests in view · {}",
            st.rows.len(),
            Local::now().format("%H:%M:%S")
        )
    } else {
        format!("monitor unreachable: {}", st.err)
    };
    let status = Line::from(vec![
        Span::styled("● ", Style::default().fg(dotc)),
        Span::styled(left, Style::default().fg(DIM)),
        Span::raw("   "),
        Span::styled("q quit · r refresh", Style::default().fg(DIM)),
    ]);
    f.render_widget(Paragraph::new(status), chunks[6]);
}

// -------------------------------------------------------------------- main --

fn main() {
    // --frames N: exit after N polls (for capture/testing)
    let args: Vec<String> = std::env::args().collect();
    let mut frames: Option<u32> = None;
    if let Some(i) = args.iter().position(|a| a == "--frames") {
        if i + 1 < args.len() {
            frames = args[i + 1].parse().ok();
        }
    }
    let mut terminal =
        Terminal::new(CrosstermBackend::new(std::io::stdout())).expect("init terminal");
    std::io::stdout().execute(EnterAlternateScreen).ok();
    std::io::stdout().execute(Hide).ok();
    enable_raw_mode().expect("raw mode");
    let mut last = Stats::default();
    let mut next_fetch = Instant::now();
    let mut polls = 0u32;
    loop {
        if let Some(n) = frames {
            if polls >= n {
                break;
            }
        }
        polls += 1;
        terminal.draw(|f| draw(f, &last)).expect("draw");
        if event::poll(Duration::from_millis(0)).unwrap_or(false) {
            match event::read() {
                Ok(Event::Key(k)) => match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    _ => {}
                },
                _ => {}
            }
        }
        if Instant::now() >= next_fetch {
            next_fetch = Instant::now() + Duration::from_secs(1);
            match fetch() {
                Ok(v) => last = Stats::parse(&v),
                Err(e) => {
                    last.ok = false;
                    last.err = e;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    disable_raw_mode().ok();
    std::io::stdout().execute(Show).ok();
    std::io::stdout().execute(LeaveAlternateScreen).ok();
}
