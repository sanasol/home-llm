# home-llm

Public build log from my home LLM setup in Belgrade: first a Steam Deck + eGPU, then a
3-GPU desktop that ended up serving everything locally. All numbers are measured here —
bench harnesses, raw JSON and configs are in this repo so you can re-run them.

```
Phase 0 (Oct)      Mac M3 Max 48 GB -> 5 engines, 27B model, 20K/107K agent chats
Phase 1 (25-26.10) Steam Deck OLED + DEG1 dock + RTX 5090 (PCIe 3.0 x4), Ubuntu on microSD
Phase 2 (26.10+)   3-GPU rig: RTX 5090 + 2x RTX 3090 (one on OCuLink x4), Ryzen 9 9950X, 96 GB DDR5
```

**Hardware line** (rig): Ryzen 9 9950X (16C/32T, AVX-512 VNNI), 96 GB DDR5-6000 (2 channels),
Samsung 990 PRO 2 TB, Linux (Ubuntu 26.10), NVIDIA driver 615.71, CUDA 13.1.
RTX 5090 32 GB (PCIe 5.0 x8) + RTX 3090 24 GB (PCIe 4.0 x4, OCuLink) + RTX 3090 24 GB (PCIe 4.0 x8).

## Headline numbers

| Phase | Setup | Result |
|---|---|---|
| Mac | Qwen3.8-27B, M3 Max 48 GB, same 20K-token agent chat | Splash+DFlash2 **33.4 t/s** decode; TensorFold 26.7; oMLX 23-26; llama.cpp 12.7 |
| Deck | RTX 5090 over PCIe 3.0 x4, Qwen3.8-27B NVFP4, NInfer | **11.3k t/s prefill**, ~257 t/s decode, 240K context with vision |
| Rig 27B | 2x RTX 3090, no NVLink, vLLM | TP=2 **6.5 t/s** -> PP=2 **44.6 t/s** -> one engine per card **95-100 t/s** |
| Rig 27B | fan fix | blower floor **41% -> 0%** idle via driver runtime override (Ampere Cert 3.0 blocks flashed VBIOS) |
| Rig Flash-Next | 5090 + 2x 3090, one 50 GB expert arena in RAM | **371-392 t/s aggregate**, 4 streams, 512K context on the 5090 |
| Rig Flash-Next | Cyrillic draft vocab | Russian **151 -> 212 t/s** on 5090, acceptance 42% -> 69% |
| Rig Flash-Next | disk/RAM conversation parking | 442K-token conversation back in **4.8 s** vs 74 s cold |
| Rig DeepSeek | whole box on one model (5090 + 2x 3090) | DSpark **48 -> 68 t/s** decode; 483K tokens read in 7.5 min |

## What's here

| Path | What |
|---|---|
| [`deck/`](deck/README.md) | Steam Deck as an LLM server: services, launch order, pitfalls; monitor + TUI code |
| [`bench/results/`](bench/results/) | Deck phase raw benchmarks (TensorFold vs NInfer, KV dtype/draft sweep), full JSONs |
| [`mac-local/`](mac-local/) | Mac phase harnesses and raw results (long-chat, agent bench) |
| [`rig/`](rig/) | rig configs (systemd units, Strata presets with redacted keys), monitor/Grafana, LiteLLM hook, patches, bench scripts |
| [`docs/journal.md`](docs/journal.md) | dated build log, milestone by milestone, with the numbers above |
| [`install/`](install/) | autoinstall templates (no keys: `@SSH_KEYS@` placeholder) |

## Timeline

- **10.10** Mac M3 Max 48 GB: TensorFold 0.6.6 + Qwen3.8-27B 4-bit + DFlash2, 128K context inside a 34 GiB budget.
- **11.10** long-chat bench, 5 engines, same 20K-token agent chat: Splash 33.4 t/s, TensorFold 26.7, oMLX 23-26, llama.cpp MTP 12.7. Prefill, not decode, decides agent latency.
- **25-26.10** Steam Deck OLED + Minisforum DEG1 eGPU dock + RTX 5090: NInfer (Qwen3.8-27B NVFP4 + DFlash2 + vision), 11.3k t/s prefill, KV fp8 -> k8v4 (+28% KV pool), 240K context with vision.
- **26.10** rig assembled (Ubuntu 26.10 headless over Wi-Fi, rescue ISO).
- **26.10** 2x RTX 3090 without P2P: vLLM TP=2 6.5 t/s -> PP=2 44.6 t/s -> one engine per card 95-100 t/s.
- **26-27.10** blower 3090 fan floor 41% -> 0% idle with a runtime override (Cert 3.0 prevents flashing a modded VBIOS).
- **26-27.10** web monitor (:8090) with SQLite history; **27.10** LiteLLM single endpoint, Prometheus/Grafana and Telegram alerts, external watchdog.
- **27.10** GPU-unlock: BF16 GEMM **103.4 -> 66.5 TFLOPS (+55%)**, decode/prefill unchanged.
- **28-29.10** OpenJet trial: synthetic bench said 2.6x faster, real agent task **531 s vs 116 s** -> rolled back same day.
- **29.10** Russian glitches in Qwen3.8 long answers: 18.3% -> **1.7%** by sampling (T 0.7 + min_p 0.05), not a new model.
- **29-30.10** Cyrillic draft vocab for MTP: RU 55 -> 101.5 t/s on one 3090 (upstream HyperQwen PR #423 merged), same trick in Strata: 151 -> 212 t/s on 5090.
- **30.10** Qwen3.8-Flash-Next on all three cards: one copy of the experts in a 50 GB shared RAM arena, 371-392 t/s aggregate, 512K context on the 5090.
- **01-02.11** abliterated model on all lanes; conversation parking: 102K-token return in 0.9 s (vs 15 s re-prefill), 442K-token return in 4.8 s (vs 74 s cold).
- **06.11** Strata 0.2.11's DFlash2-auto silently broke session saves; wrote the fix that became upstream #1492.
- **10-12.11** DFlash2 on streaming cards: always worse than MTP for Russian, two rounds; layer-major prefill is not our case (experts live in RAM).
- **12.11** DeepSeek-V4.1-Flash (Maya) on the whole box: DSpark decode 48 -> 68 t/s, cold prefill x1.9-2.1, 483K tokens read in 7.5 min, layer split tuned to link speed (x4 card gets fewer layers).

## Reproduce

Short version: [`docs/journal.md`](docs/journal.md) links the phase and the number; the harness and
raw result are under [`bench/`](bench/) or [`rig/scripts/`](rig/scripts/). No LICENSE file is added;
upstream patches and vendored assets keep their own licenses (see `rig/patches/README.md`).
