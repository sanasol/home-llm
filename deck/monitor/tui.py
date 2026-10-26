"""Terminal dashboard for the NInfer monitor on the Steam Deck.

Reads /api/state from ninfer-monitor (:8090) once a second and renders it with
curses: big decode/prefill/kv numbers (7-row pixel font) with sparklines,
gauge bars for GPU/VRAM/CPU/RAM/disk, request queue, KV pool, prompt cache,
draft acceptance, TTFT, model-weight load progress, and the last finished
requests. Keys: q — quit, r — refresh now.

Env: MONITOR_URL (http://127.0.0.1:8090).
"""
import curses
import json
import os
import re
import sys
import time
import urllib.request

URL = os.environ.get("MONITOR_URL", "http://127.0.0.1:8090")
POLL_S = 1.0
_last_err = ""
MODEL_RE = re.compile(r'model_name="([^"]+)"')
SPEC_RE = re.compile(r'speculative_backend="([^"]+)"')
SPARK = "▁▂▃▄▅▆▇█"
# 5x7 pixel font for the big numbers (bit 4 = leftmost column)
FONT = {
    "0": (14, 17, 19, 21, 25, 17, 14),
    "1": (4, 12, 4, 4, 4, 4, 14),
    "2": (14, 17, 1, 6, 8, 16, 31),
    "3": (14, 17, 1, 7, 1, 17, 14),
    "4": (2, 6, 10, 18, 31, 2, 2),
    "5": (31, 16, 30, 1, 1, 17, 14),
    "6": (6, 8, 16, 30, 17, 17, 14),
    "7": (31, 1, 1, 4, 8, 8, 8),
    "8": (14, 17, 17, 14, 17, 17, 14),
    "9": (14, 17, 17, 15, 1, 2, 12),
    ".": (0, 0, 0, 0, 0, 0, 14),
    "k": (8, 10, 12, 8, 12, 10, 10),
    "-": (0, 0, 0, 31, 0, 0, 0),
}


def fetch():
    req = urllib.request.Request(URL + "/api/state")
    with urllib.request.urlopen(req, timeout=4) as r:
        return json.loads(r.read())


def f(v, d=0):
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    try:
        return f"{v:,.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def gib(b):
    return b / 1073741824


def bar(pct, w):
    """A text progress bar; pct may be None."""
    if pct is None:
        return "-" * w
    n = max(0, min(w, round(w * pct / 100)))
    return "#" * n + "." * (w - n)


def big_val(v, d=1):
    """Format a value for the big number blocks."""
    if v is None:
        return "-"
    if v >= 1000:
        return f"{v / 1000:.2f}k"
    return f"{v:.{d}f}"


def spark(vals, w):
    """A one-line block sparkline over the last w non-null values."""
    vals = [v for v in vals if v is not None][-w:]
    if len(vals) < 3:
        return "·" * w
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1.0
    return "".join(SPARK[min(7, int((v - lo) / rng * 7.999))] for v in vals).rjust(w)


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    curses.use_default_colors()
    try:
        curses.init_pair(1, curses.COLOR_GREEN, -1)   # ok
        curses.init_pair(2, curses.COLOR_YELLOW, -1)  # warn
        curses.init_pair(3, curses.COLOR_RED, -1)     # bad
        curses.init_pair(4, curses.COLOR_CYAN, -1)    # accent
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)
        curses.init_pair(6, curses.COLOR_BLUE, -1)
    except curses.error:
        pass
    global _last_err
    while True:
        ch = stdscr.getch()
        if ch == ord("q"):
            break
        t0 = time.time()
        try:
            s = fetch()
            _last_err = ""
        except Exception as e:
            _last_err = repr(e)
            time.sleep(max(0.0, POLL_S - (time.time() - t0)))
            continue
        draw(stdscr, s)
        if ch == ord("r"):  # pressed right after the last read: refresh again this tick
            time.sleep(0.2)
            continue
        time.sleep(max(0.0, POLL_S - (time.time() - t0)))
    return 0


def put(scr, y, x, text, attr=0):
    h, w = scr.getmaxyx()
    if y >= h or x >= w:
        return
    try:  # writing the bottom-right cell can still fail; rendering must never crash
        scr.addnstr(y, x, text, w - 1 - x, attr)
    except curses.error:
        pass


def big(scr, y, x, text, attr):
    """Render text with the 5x7 pixel font, 7 rows tall, one cell per pixel."""
    for r in range(7):
        line = []
        for ch in text:
            g = FONT.get(ch)
            if g is None:
                line.append(" " * 6)
                continue
            bits = g[r]
            line.append("".join("█" if bits & (16 >> c) else " " for c in range(5)) + " ")
        put(scr, y + r, x, "".join(line).rstrip(), attr)


def kv_row(scr, y, label, value, sub="", pct=None, color=0, x0=1):
    """One LLM metric row: 'label bar value  sub'; x0 = column offset."""
    put(scr, y, x0, f"{label:<14}", curses.A_DIM)
    put(scr, y, x0 + 13, bar(pct, 10) + " ", color)
    put(scr, y, x0 + 24, f"{value:<16}", curses.A_BOLD | color)
    if sub:
        put(scr, y, x0 + 41, sub, curses.A_DIM)


def hw_row(scr, y, label, pct, value, sub="", color=0, x0=1):
    """One hardware gauge row: 'label [████████░░░░░░░░] value  sub'."""
    put(scr, y, x0, f"{label:<14}", curses.A_DIM)
    if pct is None:
        put(scr, y, x0 + 14, "-" * 18, curses.A_DIM)
    else:
        n = int(round(16 * max(0.0, min(100.0, pct)) / 100))
        put(scr, y, x0 + 14, "[" + "█" * n + "░" * (16 - n) + "]", color)
    put(scr, y, x0 + 33, f"{value:<16}", curses.A_BOLD | color)
    if sub:
        put(scr, y, x0 + 50, sub, curses.A_DIM)


def load_row(scr, y, s):
    """Model-weight load progress; returns True while something is shown."""
    ld = s.get("model_load") or {}
    if ld.get("phase") == "loading":
        pct = ld.get("pct") or 0.0
        n = int(round(24 * pct / 100))
        put(scr, y, 1, "LOADING WEIGHTS", curses.A_BOLD | curses.color_pair(2))
        put(scr, y, 18, "[" + "█" * n + "░" * (24 - n) + "]", curses.color_pair(2))
        put(scr, y, 44, f"{pct:5.1f}%", curses.A_BOLD | curses.color_pair(2))
        put(scr, y, 54, f"{ld.get('loaded_gib') or 0:.1f}/{ld.get('total_gib') or 0:.1f} GiB · "
            f"{ld.get('speed_mib_s') or 0:.0f} MiB/s · ETA {ld.get('eta_text') or '—'}", curses.color_pair(2))
        return True
    if ld.get("phase") == "finalizing":
        put(scr, y, 1, "WEIGHTS READY — finalizing: CUDA graphs, host context",
            curses.A_BOLD | curses.color_pair(1))
        return True
    return False


def draw(scr, s):
    m, g, hst = s["ninfer"], s["gpu"], s["host"]
    H = s.get("history") or []
    lp = H[-1] if H else {}
    ready = m.get("ninfer_engine_ready") == 1
    stale = time.time() - s.get("updated", 0) > 5
    info = next((k for k in m if k.startswith("ninfer_model_info")), "")
    m1 = MODEL_RE.search(info) if info else None
    model = m1.group(1) if m1 else "?"
    spec_m = SPEC_RE.search(info) if info else None
    spec = spec_m.group(1) if spec_m else ""
    ctx = f" · {m['ninfer_max_context_tokens'] // 1024:.0f}K ctx" if m.get("ninfer_max_context_tokens") else ""

    # header
    svc_txt = "STALE" if stale else ("READY" if ready else ("LOADING" if s.get("service") in ("active", "activating") else "DOWN"))
    svc_attr = {
        "STALE": curses.color_pair(3), "READY": curses.color_pair(1),
        "LOADING": curses.color_pair(2), "DOWN": curses.color_pair(3),
    }[svc_txt]
    put(scr, 0, 1, "DECK 5090 · NINFER", curses.A_BOLD | curses.color_pair(4))
    put(scr, 0, 22, f"{model} · {spec}{ctx}", curses.color_pair(5))
    put(scr, 0, 62, f"[{svc_txt}]", svc_attr | curses.A_BOLD)
    put(scr, 0, 70, time.strftime("%H:%M:%S"), curses.A_DIM)
    put(scr, 1, 1, "q quit · r refresh", curses.A_DIM)

    run = m.get("ninfer_requests_running")
    wait = m.get("ninfer_requests_waiting")
    paused = m.get("ninfer_requests_paused")
    dec, pre = lp.get("decode_tps"), lp.get("prefill_tps")
    kv_u = m.get('ninfer_device_kv_used_page_groups{pool_id="0"}')
    kv_c = m.get('ninfer_device_kv_capacity_page_groups{pool_id="0"}')
    kv_pct = 100 * kv_u / kv_c if kv_c else None
    cap_tok = s.get("kv_capacity_tokens")
    kv_tok = cap_tok * kv_u / kv_c if kv_c and cap_tok else None
    prompt = m.get("ninfer_prompt_tokens_total")
    cached = m.get("ninfer_prompt_tokens_cached_total")
    hits = 100 * cached / prompt if prompt else None
    drafted = m.get("ninfer_spec_decode_draft_tokens_total")
    acc = 100 * m.get("ninfer_spec_decode_accepted_tokens_total", 0) / drafted if drafted else None
    ttft_n = m.get("ninfer_time_to_first_token_seconds_count")
    ttft = m.get("ninfer_time_to_first_token_seconds_sum", 0) / ttft_n if ttft_n else None
    done = m.get('ninfer_requests_total{outcome="completed"}')
    failed = (m.get('ninfer_requests_total{outcome="failed"}') or 0) + (m.get('ninfer_requests_total{outcome="rejected"}') or 0)
    mem_used = hst.get("mem_total", 0) - hst.get("mem_available", 0)
    mem_pct = 100 * mem_used / hst["mem_total"] if hst.get("mem_total") else None
    dsk_tot, dsk_av = hst.get("disk_total"), hst.get("disk_avail")
    dsk_pct = 100 * (dsk_tot - dsk_av) / dsk_tot if dsk_tot else None
    vram_pct = 100 * g["memory.used"] / g["memory.total"] if g.get("memory.total") else None

    two = scr.getmaxyx()[1] >= 116

    # big number blocks: decode / prefill / (kv) + sparklines, then the load progress row
    big(scr, 3, 2, big_val(dec), curses.color_pair(1) if dec else curses.A_DIM)
    big(scr, 3, 54, big_val(pre), curses.color_pair(5) if pre else curses.A_DIM)
    put(scr, 10, 2, "DECODE tok/s", curses.A_DIM)
    put(scr, 10, 54, "PREFILL tok/s", curses.A_DIM)
    put(scr, 11, 2, spark([p.get("decode_tps") for p in H], 29), curses.color_pair(1))
    put(scr, 11, 54, spark([p.get("prefill_tps") for p in H], 29), curses.color_pair(5))
    if two:
        big(scr, 3, 106, big_val(kv_pct, 0), curses.color_pair(4) if kv_pct is not None else curses.A_DIM)
        put(scr, 10, 106, "KV POOL %", curses.A_DIM)
        put(scr, 11, 106, bar(kv_pct, 29), curses.color_pair(4))
    load_row(scr, 12, s)

    # LLM section (left column)
    put(scr, 13, 1, "LLM", curses.A_BOLD | curses.color_pair(4))
    kv_row(scr, 14, "requests", f"{f(run)} / {f(m.get('ninfer_max_concurrency'))}",
           f"wait {f(wait)} · paused {f(paused)}",
           100 * run / m["ninfer_max_concurrency"] if m.get("ninfer_max_concurrency") else None,
           curses.color_pair(1) if run else 0)
    kv_row(scr, 15, "KV pool", f"{f(kv_pct, 0)}%",
           f"≈{f(kv_tok / 1000)}K / {f(cap_tok / 1000)}K tok" if kv_tok else "", kv_pct)
    kv_row(scr, 16, "cache hits", f"{f(hits, 1)}%", f"{f(cached)} / {f(prompt)} tok")
    kv_row(scr, 17, "draft accept", f"{f(acc, 1)}%", f"{f(m.get('ninfer_spec_decode_accepted_tokens_total'))} tok")
    kv_row(scr, 18, "TTFT avg", f"{f(ttft, 2)} s", f"{f(done)} done · {f(failed)} fail")

    # HARDWARE section: gauges, right column when wide, below the LLM section otherwise
    if two:
        hw_y0 = 14
        put(scr, 13, 66, "HARDWARE", curses.A_BOLD | curses.color_pair(4))
    else:
        hw_y0 = 20
        put(scr, 19, 1, "HARDWARE", curses.A_BOLD | curses.color_pair(4))
    gpu = g.get("utilization.gpu")
    hw_row(scr, hw_y0, "GPU", gpu, f"{f(gpu)}%",
           f"{f(g.get('power.draw'))} / {f(g.get('power.limit'))} W · {f(g.get('temperature.gpu'))} °C",
           curses.color_pair(2) if (gpu or 0) >= 90 else curses.color_pair(1), 66 if two else 1)
    # nvidia-smi reports memory in MiB, not bytes
    hw_row(scr, hw_y0 + 1, "VRAM", vram_pct,
           f"{f(g.get('memory.used', 0) / 1024, 1)} / {f(g.get('memory.total', 0) / 1024, 1)} GiB",
           f"PCIe {f(g.get('pcie.link.gen.current'))}.0 x{f(g.get('pcie.link.width.current'))}",
           curses.color_pair(3) if (vram_pct or 0) >= 95 else 0, 66 if two else 1)
    cpu = lp.get("cpu_pct")
    hw_row(scr, hw_y0 + 2, "CPU", cpu, f"{f(cpu, 1)}%", f"load {f(hst.get('load1'), 2)}",
           curses.color_pair(2) if (cpu or 0) >= 80 else 0, 66 if two else 1)
    hw_row(scr, hw_y0 + 3, "RAM", mem_pct,
           f"{f(gib(mem_used), 1)} / {f(gib(hst.get('mem_total')), 1)} GiB",
           f"avail {f(gib(hst.get('mem_available')), 1)} GiB", 0, 66 if two else 1)
    hw_row(scr, hw_y0 + 4, "disk", dsk_pct,
           f"{f(gib(dsk_tot - dsk_av), 0)} / {f(gib(dsk_tot), 0)} GiB",
           f"r {f(lp.get('disk_r_mbps'), 2)} · w {f(lp.get('disk_w_mbps'), 2)} MB/s", 0, 66 if two else 1)

    # recent requests table
    y = 20 if two else 26
    put(scr, y, 1, "RECENT REQUESTS", curses.A_BOLD | curses.color_pair(4))
    hdr = f"{'time':<8} {'api':<12} {'prompt':>8} {'cached':>7} {'out':>7} {'ttft':>7} {'pre t/s':>8} {'dec t/s':>8} {'acc':>6} {'total':>7}"
    put(scr, y + 1, 1, hdr, curses.A_DIM)
    for i, r in enumerate((s.get("requests") or [])[:9]):
        put(scr, y + 2 + i, 1,
            f"{r['ts'][11:]:<8} {r['api']:<12} {f(r['prompt']):>8} {f(r['cache_pct'], 0)}% {f(r['output']):>7} "
            f"{f(r['ttft'], 2):>7} {f(r.get('prefill_tps')):>8} {f(r.get('decode_tps')):>8} "
            f"{('-' if r.get('accept_pct') is None else f(r['accept_pct'], 0) + '%'):>6} {f(r['total'], 1):>7}")
    if not s.get("requests"):
        put(scr, y + 2, 1, "no requests yet", curses.A_DIM)

    if _last_err:
        put(scr, 1, 40, f"monitor unreachable: {_last_err[:40]}", curses.color_pair(3))
    scr.refresh()


if __name__ == "__main__":
    try:
        rc = curses.wrapper(main)
    except KeyboardInterrupt:
        rc = 0
    sys.exit(rc)
