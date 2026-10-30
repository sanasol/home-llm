# home-llm

Public artifacts from my home LLM build (Belgrade, 3 GPUs: RTX 5090 + 2× RTX 3090,
Ubuntu, all local). This repo holds the **stopgap-server phase**: running Qwen3.8-27B
on a **Steam Deck OLED + Minisforum DEG1 eGPU dock + RTX 5090** while the main rig
was still in boxes — plus the engine benchmarks, web monitor and on-screen TUI I
built for it.

```
Steam Deck OLED → M.2→OCuLink adapter → Minisforum DEG1 (own PSU) → RTX 5090
Ubuntu 26.04 on a 256 GB microSD · engine: NInfer (Qwen3.8-27B NVFP4 + DFlash2 + vision)
```

## What's here

| Path | What |
|---|---|
| [`deck/`](deck/README.md) | The Deck-as-LLM-server build: services, launch order, pitfalls; monitor + TUI code |
| [`deck/monitor/`](deck/monitor/) | Web monitor (:8090) scraping NInfer `/metrics`, `nvidia-smi` and the engine journal |
| [`deck/tui-rs/`](deck/tui-rs/) | Ratatui (Rust) TUI rendered on the Deck's built-in screen (fbcon 160×50, 8 colors) |
| [`bench/results/2026-10-25-.../`](bench/results/2026-10-25-deck-5090-tensorfold-vs-ninfer/README.md) | TensorFold vs NInfer on Deck+5090: decode vs prefill tradeoff, full JSONs |
| [`bench/results/2026-10-26-.../`](bench/results/2026-10-26-deck-kv-and-draft-sweep/README.md) | NInfer KV dtype (fp8 vs k8v4) and DFlash2 draft-window sweep; 240K context, needle tests |

## Headline numbers (Steam Deck OLED + DEG1 + RTX 5090, PCIe 3.0 x4, no ReBAR)

- Qwen3.8-27B NVFP4 (21.7 GiB weights) served by NInfer, 2 streams, vision, up to 240K/request.
- TensorFold + DFlash2: up to **410 t/s** decode on short code — but 2.6–2.8× slower prefill.
- NInfer: **11.3k t/s prefill**, ~257 t/s code decode; wins for agent traffic.
- Switching KV cache fp8 → k8v4: KV pool +28%, 240K fits with vision, no speed loss.
- The narrow OCuLink x4 link barely costs anything: decode matches full-desktop runs.

Companion rig repos and the cyrillic spec-decode work: see my profile.
