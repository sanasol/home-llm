# Steam Deck + DEG1 + RTX 5090: stopgap LLM server

While the main rig wasn't ready, the 5090 sat in a Minisforum DEG1 dock attached to a
**Steam Deck OLED** via an M.2→OCuLink adapter in the SSD slot. Ubuntu 26.04 on a
256 GB microSD, host `deck-llm`. Engine comparisons:
[bench/results/2026-10-25-deck-5090-tensorfold-vs-ninfer](../bench/results/2026-10-25-deck-5090-tensorfold-vs-ninfer/README.md).

## Bring-up order

1. Everything off, DEG1 PSU off.
2. Power the DEG1 first, then boot the Deck **from cold** and boot from SD
   (volume-down + power).
3. **Never `reboot` with the GPU attached.** Rebooting with the card connected puts
   the Deck into a boot loop (black screen, fan at 100%); the only way out is
   disconnecting the battery.

If the Deck boots without the card, `ninfer.service` just restarts every 30 s until
the GPU appears.

## Services

| Service | Port | What |
|---|---|---|
| `ninfer.service` | 8080 | [NInfer](https://github.com/Neroued/ninfer) `64d10fbf`, Qwen3.8-27B NVFP4 + DFlash2 (window 7) + **vision**, up to 240K/request, KV **k8v4** `auto` (~277K tokens across both streams; the fp8→k8v4 move and the K=7 choice are justified in [bench/results/2026-10-26-deck-kv-and-draft-sweep](../bench/results/2026-10-26-deck-kv-and-draft-sweep/README.md)), 2 streams, `--host-context-mib 4096`, `--default-max-tokens 32768` (the stock 8192 silently truncated agent context-compression summaries that run with thinking on and no explicit `max_tokens`), GPU power limit 460 W. Requires an API key (`/etc/ninfer.env`). |
| `ninfer-monitor.service` | 8090 | [`monitor/`](monitor/): NInfer `/metrics`, `nvidia-smi`, CPU, RAM, disk (IO + free), current model, recent requests parsed from the journal. JSON at `/api/state`. |

Unit files: [`ninfer.service`](ninfer.service), [`monitor/ninfer-monitor.service`](monitor/ninfer-monitor.service).
Model load from microSD takes ~4.5 minutes (~85 MiB/s).

Addresses: `http://deck-llm.local:8080/v1` and `http://deck-llm.local:8090` on LAN.

## TUI on the Deck's own screen

`tui` — a **Ratatui (Rust)** dashboard ([`tui-rs/`](tui-rs/), static binary, built on
the Deck with `cargo build --release`). A Python version is kept as `tui-py`.
Both read `http://127.0.0.1:8090/api/state` once a second: header with model +
status (READY/LOADING/STALE/DOWN), big pixel-digit decode/prefill/KV numbers with
sparklines, LLM panels (requests, KV, cache, draft-accept, TTFT) and HARDWARE panels
(GPU/VRAM/CPU/RAM/disk) with gauge bars, a recent-requests table, and a weight-load
progress line (%, GiB, MiB/s, ETA — parsed from the NInfer journal).
`q`/Esc quits, `r` refreshes. No console autostart — launch it by hand.

Screen constraints worth knowing: the built-in panel's console is fbcon 160×50,
8 colors + bold only (no 256/truecolor). Don't trust `/dev/fb0` dumps: the buffer is
portrait (stride 3328, rotated 90°) and fb0 can be the HDMI output.

## Pitfalls I hit (each one cost an evening)

- **Boot loop** on reboot with the GPU attached → see bring-up order above.
- **Pixel font bug**: the 5x7 FONT dict is bit-packed (bit 4 = leftmost column); a
  glyph edit shipped a *mirrored "1"* to production. Dump and eyeball every digit
  after editing.
- **`ETXTBSY`**: you can't `cp` over the running TUI binary — stage as
  `ninfer-tui.new` then `mv -f` (atomic rename; the running process keeps the old inode).
- **uutils coreutils** on the Deck: `/usr/bin/timeout` is Rust, not GNU.
- **No rsync** on the Deck; deploy with `scp -rq`.
- **Context ceiling**: with vision + DFlash2 + 2 streams on fp8 KV, ~216K/request
  max; an agent session hit 212,877 of the 212,992-token window and died. k8v4 KV
  lifts it to 240K (scheduler fits 262K); keep agent session peaks under ~160K.
- NInfer rejects a global `high` reasoning effort for this chat template
  (`conflicting_template_option` 400) — pin effort per model.
